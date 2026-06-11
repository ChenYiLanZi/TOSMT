import os
import random
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    pd = None
    np = None


class SyntheticMultiBehaviorDataset(Dataset):
    def __init__(self, num_items: int, seq_len: int, size: int = 1024):
        super().__init__()
        self.num_items = num_items
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create random sequences; use the same id as mock image/text token for simplicity
        click_seq = torch.randint(low=1, high=self.num_items, size=(self.seq_len,), dtype=torch.long)
        favor_seq = torch.randint(low=1, high=self.num_items, size=(self.seq_len,), dtype=torch.long)

        sample = {
            'click_id_seq': click_seq,
            'click_img_seq': click_seq.clone(),
            'click_txt_seq': click_seq.clone(),
            'favor_id_seq': favor_seq,
            'favor_img_seq': favor_seq.clone(),
            'favor_txt_seq': favor_seq.clone(),
            'labels': torch.randint(low=0, high=self.num_items, size=(1,), dtype=torch.long).squeeze(0),
        }
        return sample


class KuaiLiveDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        seq_len: int,
        min_len: int = 5,
        min_item_freq: int = 10,
        min_user_freq: int = 5,
        data_protocol: str = 'mmsr_loo_fixedneg',
        item_id_source: str = 'auto',
        hard_neg_pool_size: int = 1500,
        target_behaviors: Optional[str] = None,
        auxiliary_behaviors: Optional[str] = None,
        csv_source: str = 'auto',
    ):
        super().__init__()
        if pd is None or np is None:
            raise RuntimeError("pandas/numpy 未安装，无法解析 KuaiLive。")
        self.data_root = data_root
        self.seq_len = seq_len
        self.min_len = min_len
        self.min_item_freq = min_item_freq  # 因高频数据过滤：添加item最小频率阈值
        self.min_user_freq = min_user_freq  # 因高频数据过滤：添加user最小频率阈值
        self.data_protocol = data_protocol
        self.hard_neg_pool_size = hard_neg_pool_size
        if target_behaviors:
            self.target_behavior_set = {b.strip().lower() for b in target_behaviors.split(',') if b.strip()}
        else:
            self.target_behavior_set = {'like', 'favorite', 'fav'}
        if auxiliary_behaviors:
            self.auxiliary_behavior_set = {b.strip().lower() for b in auxiliary_behaviors.split(',') if b.strip()}
        else:
            self.auxiliary_behavior_set = set()
        self.csv_source = csv_source
        self.samples: List[Dict[str, Any]] = []
        self.split_indices: Dict[str, List[int]] = {}
        self.num_items: int = 0
        self.requested_item_id_source = item_id_source
        self.item_id_source = 'streamer_id' if item_id_source == 'auto' else item_id_source
        self.item2idx: Dict[str, int] = {}
        self._prepare()

    def _find_csv_files(self) -> List[str]:
        behavior_files = {'comment.csv', 'gift.csv', 'like.csv'}
        if self.csv_source == 'behavior_files':
            files = [
                os.path.join(self.data_root, fn)
                for fn in os.listdir(self.data_root)
                if fn.lower() in behavior_files
            ]
            return sorted(files)

        action_files = {
            'df_action_final.csv',
            'userbehavior.csv',
            'user_behavior.csv',
            'behavior.csv',
            'actions.csv',
            'action.csv',
        }
        action_preferred = []
        for fn in os.listdir(self.data_root):
            if fn.lower() in action_files:
                action_preferred.append(os.path.join(self.data_root, fn))
        if action_preferred:
            return sorted(action_preferred)

        preferred = []
        for fn in os.listdir(self.data_root):
            if fn.lower() in behavior_files:
                preferred.append(os.path.join(self.data_root, fn))
        if preferred:
            return sorted(preferred)

        files = []
        for fn in os.listdir(self.data_root):
            lower = fn.lower()
            if lower.endswith('.csv') and 'mapping' not in lower:
                files.append(os.path.join(self.data_root, fn))
        return sorted(files)

    def _standardize_columns(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        cols = {c.lower(): c for c in df.columns}
        # 适配KuaiLive数据格式：comment.csv, like.csv, gift.csv
        user_col = cols.get('user_id') or cols.get('uid') or list(df.columns)[0]
        
        if self.requested_item_id_source != 'auto' and self.requested_item_id_source in cols:
            item_col = self.requested_item_id_source
            self.item_id_source = self.requested_item_id_source
        # Prefer streamer_id by default: the provided title_embeddings.npy covers streamer ids,
        # while live_id may require live/video-level text embeddings to avoid many missing rows.
        elif 'streamer_id' in cols:
            item_col = 'streamer_id'
            self.item_id_source = 'streamer_id'
        elif 'live_id' in cols:
            item_col = 'live_id'
            self.item_id_source = 'live_id'
        else:
            item_col = list(df.columns)[1]
            self.item_id_source = item_col
        
        time_col = cols.get('timestamp') or cols.get('time') or cols.get('ts') or cols.get('date')
        behavior_col = (
            cols.get('behavior')
            or cols.get('behavior_type')
            or cols.get('event_type')
            or cols.get('action')
            or cols.get('action_type')
            or cols.get('type')
            or cols.get('beh')
        )
        
        rename_map = {}
        if user_col != 'user_id':
            rename_map[user_col] = 'user_id'
        if item_col != 'item_id':
            rename_map[item_col] = 'item_id'
        if time_col and time_col != 'timestamp':
            rename_map[time_col] = 'timestamp'
        if behavior_col and behavior_col != 'behavior':
            rename_map[behavior_col] = 'behavior'
        
        # 根据文件名添加behavior字段
        filename = getattr(self, '_current_file', '')
        if behavior_col is None and 'behavior' not in df.columns and 'comment' in filename.lower():
            df['behavior'] = 'comment'
        elif behavior_col is None and 'behavior' not in df.columns and 'like' in filename.lower():
            df['behavior'] = 'like'
        elif behavior_col is None and 'behavior' not in df.columns and 'gift' in filename.lower():
            df['behavior'] = 'gift'
        elif behavior_col is None and 'behavior' not in df.columns and 'buy' in filename.lower():
            df['behavior'] = 'buy'
        elif behavior_col is None and 'behavior' not in df.columns and 'cart' in filename.lower():
            df['behavior'] = 'cart'
        elif behavior_col is None and 'behavior' not in df.columns and ('fav' in filename.lower() or 'collect' in filename.lower()):
            df['behavior'] = 'fav'
        elif behavior_col is None and 'behavior' not in df.columns and ('pv' in filename.lower() or 'view' in filename.lower() or 'click' in filename.lower()):
            df['behavior'] = 'pv'
        elif behavior_col is None and 'behavior' not in df.columns:
            df['behavior'] = 'interaction'  # 默认行为类型
        
        df = df.rename(columns=rename_map)
        return df

    def _load_merge(self) -> 'pd.DataFrame':
        csvs = self._find_csv_files()
        if not csvs:
            raise FileNotFoundError(f"KuaiLive 目录 {self.data_root} 下未找到 CSV 文件")
        frames = []
        for path in csvs:
            try:
                # 记录当前文件名用于behavior推断
                self._current_file = os.path.basename(path)
                df = pd.read_csv(path)
                df = self._standardize_columns(df)
                frames.append(df[['user_id', 'item_id'] + (["behavior"] if 'behavior' in df.columns else []) + (["timestamp"] if 'timestamp' in df.columns else [])])
            except Exception as e:
                print(f"[Warn] 跳过文件 {path}: {e}")
                continue
        if not frames:
            raise RuntimeError("未能读取任何有效的 KuaiLive CSV 文件")
        data = pd.concat(frames, ignore_index=True)
        # Drop NA and enforce types where possible
        data = data.dropna(subset=['user_id', 'item_id'])
        print(f"[Info] 成功加载 {len(data)} 条记录，包含行为类型: {data['behavior'].unique()}")
        return data

    def _map_ids(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        # Map item ids to contiguous integers starting from 1 (0 reserved for padding)
        unique_items = data['item_id'].astype(str).unique().tolist()
        item2idx = {it: i + 1 for i, it in enumerate(unique_items)}
        data['item_id'] = data['item_id'].astype(str).map(item2idx)
        self.num_items = len(item2idx) + 1  # include padding index 0
        self.item2idx = item2idx
        return data

    def _split_behaviors(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        if 'behavior' not in data.columns:
            data['behavior'] = 'favor'  # 无行为列时，全部当作目标行为
            return data
        
        # 适配KuaiLive数据的行为映射
        def map_beh(x: str) -> str:
            xl = str(x).strip().lower()
            # KuaiLive数据的行为映射
            if xl in {'like', 'favorite', 'fav'}:
                return 'favor'  # 强信号行为
            elif xl in {'comment', 'gift'}:
                return 'click'  # 弱信号行为
            else:
                return 'click'  # 默认当作弱信号
        
        data['behavior'] = data['behavior'].apply(map_beh)
        print(f"[Info] 行为映射后: {data['behavior'].value_counts().to_dict()}")
        return data

    def _split_behaviors(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        if 'behavior' not in data.columns:
            data['behavior'] = 'favor'
            print("[Info] No behavior column found; all events are treated as target behavior.")
            return data

        def map_beh(x: str) -> str:
            xl = str(x).strip().lower()
            return 'favor' if xl in self.target_behavior_set else 'click'

        data = data.copy()
        data['behavior'] = data['behavior'].astype(str).str.strip().str.lower()
        raw_counts = data['behavior'].value_counts().to_dict()
        if self.auxiliary_behavior_set:
            allowed_behaviors = self.target_behavior_set | self.auxiliary_behavior_set
            data = data[data['behavior'].isin(allowed_behaviors)].copy()
        data['behavior'] = data['behavior'].apply(map_beh)
        print(f"[Info] Raw behavior counts: {raw_counts}")
        print(f"[Info] Target behavior set: {sorted(self.target_behavior_set)}")
        if self.auxiliary_behavior_set:
            print(f"[Info] Auxiliary behavior set: {sorted(self.auxiliary_behavior_set)}")
        print(f"[Info] Behavior mapped counts: {data['behavior'].value_counts().to_dict()}")
        return data

    def _time_sort(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        if 'timestamp' in data.columns:
            try:
                data = data.sort_values(['user_id', 'timestamp'])
            except Exception:
                data = data.sort_values(['user_id'])
        else:
            data = data.sort_values(['user_id'])
        return data

    def _filter_high_freq_data(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """因高频数据过滤：只保留高频的item和user"""
        original_len = len(data)
        print(f"[Info] 开始高频数据过滤，原始数据: {original_len} 条记录")
        
        # 过滤低频item
        if self.min_item_freq > 0:
            item_counts = data['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_freq].index.tolist()
            if len(valid_items) == 0:
                raise ValueError(f"Item过滤后没有有效item！请降低min_item_freq参数（当前={self.min_item_freq}）")
            data = data[data['item_id'].isin(valid_items)].copy()
            print(f"[Info] Item过滤 (min_freq={self.min_item_freq}): {len(item_counts)} -> {len(valid_items)} 个item，剩余 {len(data)} 条记录")
        
        # 过滤低频user
        if self.min_user_freq > 0:
            user_counts = data['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_freq].index.tolist()
            if len(valid_users) == 0:
                raise ValueError(f"User过滤后没有有效user！请降低min_user_freq参数（当前={self.min_user_freq}）")
            data = data[data['user_id'].isin(valid_users)].copy()
            print(f"[Info] User过滤 (min_freq={self.min_user_freq}): {len(user_counts)} -> {len(valid_users)} 个user，剩余 {len(data)} 条记录")
        
        filtered_len = len(data)
        if filtered_len == 0:
            raise ValueError(f"高频数据过滤后数据为空！请降低min_item_freq（当前={self.min_item_freq}）或min_user_freq（当前={self.min_user_freq}）参数")
        print(f"[Info] 高频数据过滤完成: {original_len} -> {filtered_len} 条记录 (保留 {100.0 * filtered_len / original_len:.2f}%)")
        return data

    def _build_samples(self, data: 'pd.DataFrame'):
        # 参考MGPT的数据组织方法：简化数据格式，使用统一的序列构建方式
        # 原复杂逻辑注释保留：
        # 因时间泄漏问题修正：仅使用目标交互时间之前的行为作为历史
        # 旧逻辑保留如下：
        # for user_id, df_u in data.groupby('user_id'):
        #     favor_seq = df_u[df_u['behavior'] == 'favor']['item_id'].tolist() ...
        #     ... 使用 click_seq_all 的最后片段，可能包含未来交互（注释：因时间泄漏修正而废弃）

        # 新逻辑：参考MGPT的简化数据组织方式
        user_count = 0
        total_users = len(data.groupby('user_id'))
        print(f"[Info] 开始处理 {total_users} 个用户的数据...")
        
        for user_id, df_u in data.groupby('user_id'):
            user_count += 1
            if user_count % 5000 == 0:  # 减少输出频率
                print(f"[Info] 已处理 {user_count}/{total_users} 个用户，当前样本数: {len(self.samples)}")
            df_u = df_u.copy()
            if 'timestamp' in df_u.columns:
                df_u = df_u.sort_values('timestamp')
            
            # 参考MGPT：构建统一的物品序列和行为序列
            item_sequence = []
            behavior_sequence = []
            target_items = []
            
            # 累积历史
            favor_hist: List[int] = []
            click_hist: List[int] = []
            
            for _, row in df_u.iterrows():
                item = int(row['item_id'])
                beh = row['behavior'] if 'behavior' in df_u.columns else 'favor'
                
                # 构建统一的序列（参考MGPT方式）
                item_sequence.append(item)
                behavior_sequence.append(1 if beh == 'favor' else 0)  # 1表示favor，0表示click
                
                # 参考MGPT：每5个行为预测一次下一个物品
                if len(item_sequence) % 5 == 0 and len(item_sequence) < len(df_u):
                    target_items.append(item)
                
                # 保持原有的历史累积逻辑用于多模态数据
                if beh == 'favor':
                    favor_hist.append(item)
                else:
                    click_hist.append(item)
            
            # 降低序列长度要求，允许更短的序列
            min_seq_len = max(5, self.seq_len // 2)  # 至少5个，或者seq_len的一半
            if len(item_sequence) >= min_seq_len and len(target_items) > 0:
                # 注释掉调试信息以加快处理速度
                # if user_count <= 5:  # 只对前几个用户打印调试信息
                #     print(f"[Debug] 用户 {user_id}: 序列长度={len(item_sequence)}, 目标物品数={len(target_items)}")
                # 截断或填充序列到指定长度
                if len(item_sequence) >= self.seq_len:
                    item_seq = item_sequence[-self.seq_len:]
                    behavior_seq = behavior_sequence[-self.seq_len:]
                else:
                    # 如果序列较短，用第一个元素填充
                    item_seq = item_sequence + [item_sequence[0]] * (self.seq_len - len(item_sequence))
                    behavior_seq = behavior_sequence + [behavior_sequence[0]] * (self.seq_len - len(behavior_sequence))
                
                # 为每个目标物品创建样本
                for target_item in target_items:
                    # 使用简化的数据格式（参考MGPT）
                    sample = {
                        'item_sequence': torch.tensor(item_seq, dtype=torch.long),
                        'behavior_sequence': torch.tensor(behavior_seq, dtype=torch.long),
                        'labels': torch.tensor(target_item, dtype=torch.long),
                        # 保留原有的多模态字段以兼容现有模型
                        'click_id_seq': torch.tensor(click_hist[-self.seq_len:] if len(click_hist) >= self.seq_len else click_hist, dtype=torch.long),
                        'click_img_seq': torch.tensor(click_hist[-self.seq_len:] if len(click_hist) >= self.seq_len else click_hist, dtype=torch.long),
                        'click_txt_seq': torch.tensor(click_hist[-self.seq_len:] if len(click_hist) >= self.seq_len else click_hist, dtype=torch.long),
                        'favor_id_seq': torch.tensor(favor_hist[-self.seq_len:] if len(favor_hist) >= self.seq_len else favor_hist, dtype=torch.long),
                        'favor_img_seq': torch.tensor(favor_hist[-self.seq_len:] if len(favor_hist) >= self.seq_len else favor_hist, dtype=torch.long),
                        'favor_txt_seq': torch.tensor(favor_hist[-self.seq_len:] if len(favor_hist) >= self.seq_len else favor_hist, dtype=torch.long),
                    }
                    # 过滤空行为历史样本：click_hist 和 favor_hist 都为空则跳过
                    if len(click_hist) < 1 and len(favor_hist) < 1:
                        continue
                    self.samples.append(sample)

    def _sample_fixed_negatives(self, pos_item: int, user_items: set, rng: 'np.random.Generator', num_neg: int = 99):
        pool = [item for item in range(1, self.num_items) if item != pos_item and item not in user_items]
        if not pool:
            pool = [item for item in range(1, self.num_items) if item != pos_item]
        replace = len(pool) < num_neg
        return torch.tensor(rng.choice(pool, size=num_neg, replace=replace), dtype=torch.long)

    def _make_strict_sample(self, prior_data: 'pd.DataFrame', target_item: int, negatives=None):
        item_seq = prior_data['item_id'].astype(int).tolist()[-self.seq_len:]
        behavior_seq = [1 if b == 'favor' else 0 for b in prior_data['behavior'].tolist()][-self.seq_len:]
        click_hist = prior_data[prior_data['behavior'] != 'favor']['item_id'].astype(int).tolist()[-self.seq_len:]
        favor_hist = prior_data[prior_data['behavior'] == 'favor']['item_id'].astype(int).tolist()[-self.seq_len:]

        if not item_seq:
            return None

        sample = {
            'item_sequence': torch.tensor(item_seq, dtype=torch.long),
            'behavior_sequence': torch.tensor(behavior_seq, dtype=torch.long),
            'labels': torch.tensor(int(target_item), dtype=torch.long),
            'click_id_seq': torch.tensor(click_hist, dtype=torch.long),
            'click_img_seq': torch.tensor(click_hist, dtype=torch.long),
            'click_txt_seq': torch.tensor(click_hist, dtype=torch.long),
            'favor_id_seq': torch.tensor(favor_hist, dtype=torch.long),
            'favor_img_seq': torch.tensor(favor_hist, dtype=torch.long),
            'favor_txt_seq': torch.tensor(favor_hist, dtype=torch.long),
        }
        if negatives is not None:
            sample['negatives'] = negatives
        return sample

    def _build_strict_loo_samples(self, data: 'pd.DataFrame', fixed_negatives: bool = True):
        """Build leak-free target-behavior samples.

        Each sample only sees interactions before the target behavior event.
        The last target event per user is test, the previous one is validation,
        and earlier target events are training samples.
        """
        rng = np.random.default_rng(2026)
        self.samples = []
        self.split_indices = {'train': [], 'val': [], 'test': []}
        total_users = data['user_id'].nunique()
        used_users = 0

        print(f"[Info] Building strict leave-one-out samples for {total_users} users...")
        for user_count, (_, df_u) in enumerate(data.groupby('user_id'), start=1):
            if user_count % 5000 == 0:
                print(f"[Info] Processed {user_count}/{total_users} users, samples={len(self.samples)}")

            df_u = df_u.sort_values('timestamp') if 'timestamp' in df_u.columns else df_u
            df_u = df_u.reset_index(drop=True)
            target_positions = df_u.index[df_u['behavior'] == 'favor'].tolist()
            if len(target_positions) < 2:
                continue

            used_users += 1
            user_items = set(df_u['item_id'].astype(int).tolist())
            train_targets = target_positions[:-2]
            val_target = target_positions[-2]
            test_target = target_positions[-1]

            for pos_idx in train_targets:
                sample = self._make_strict_sample(df_u.iloc[:pos_idx], int(df_u.at[pos_idx, 'item_id']))
                if sample is not None:
                    self.split_indices['train'].append(len(self.samples))
                    self.samples.append(sample)

            for split_name, pos_idx in [('val', val_target), ('test', test_target)]:
                pos_item = int(df_u.at[pos_idx, 'item_id'])
                negatives = self._sample_fixed_negatives(pos_item, user_items, rng) if fixed_negatives else None
                sample = self._make_strict_sample(df_u.iloc[:pos_idx], pos_item, negatives=negatives)
                if sample is not None:
                    self.split_indices[split_name].append(len(self.samples))
                    self.samples.append(sample)

        print(
            "[Info] Strict LOO samples built: "
            f"users={used_users}, train={len(self.split_indices['train'])}, "
            f"val={len(self.split_indices['val'])}, test={len(self.split_indices['test'])}, "
            f"total={len(self.samples)}"
        )

    def _sample_mmsr_negatives(self, pos_item: int, rng: 'np.random.Generator', num_neg: int = 99):
        pool = [item for item in range(1, self.num_items) if item != pos_item]
        replace = len(pool) < num_neg
        return torch.tensor(rng.choice(pool, size=num_neg, replace=replace), dtype=torch.long)

    def _build_mmsr_loo_samples(
        self,
        data: 'pd.DataFrame',
        fixed_negatives: bool = True,
        new_item_only: bool = False,
    ):
        """Build the leave-one-out protocol described in MMSR/M3BSR.

        For each user we keep the most recent seq_len records, use favor as
        the target behavior, use the last favor for test, the previous favor
        for validation, and earlier favor events for training. Negative items
        are randomly sampled from the item universe.
        """
        rng = np.random.default_rng(2025)
        self.samples = []
        self.split_indices = {'train': [], 'val': [], 'test': []}
        total_users = data['user_id'].nunique()
        used_users = 0

        protocol_name = "MMSR new-item leave-one-out" if new_item_only else "MMSR leave-one-out"
        print(f"[Info] Building {protocol_name} samples for {total_users} users...")
        for user_count, (_, df_u) in enumerate(data.groupby('user_id'), start=1):
            if user_count % 5000 == 0:
                print(f"[Info] Processed {user_count}/{total_users} users, samples={len(self.samples)}")

            df_u = df_u.sort_values('timestamp') if 'timestamp' in df_u.columns else df_u
            df_u = df_u.tail(self.seq_len).reset_index(drop=True)
            target_positions = df_u.index[df_u['behavior'] == 'favor'].tolist()
            if len(target_positions) < 2:
                continue

            used_users += 1
            train_targets = target_positions[:-2]
            val_target = target_positions[-2]
            test_target = target_positions[-1]

            for pos_idx in train_targets:
                if new_item_only and int(df_u.at[pos_idx, 'item_id']) in set(df_u.iloc[:pos_idx]['item_id'].astype(int).tolist()):
                    continue
                sample = self._make_strict_sample(df_u.iloc[:pos_idx], int(df_u.at[pos_idx, 'item_id']))
                if sample is not None:
                    self.split_indices['train'].append(len(self.samples))
                    self.samples.append(sample)

            for split_name, pos_idx in [('val', val_target), ('test', test_target)]:
                pos_item = int(df_u.at[pos_idx, 'item_id'])
                if new_item_only and pos_item in set(df_u.iloc[:pos_idx]['item_id'].astype(int).tolist()):
                    continue
                negatives = self._sample_mmsr_negatives(pos_item, rng) if fixed_negatives else None
                sample = self._make_strict_sample(df_u.iloc[:pos_idx], pos_item, negatives=negatives)
                if sample is not None:
                    self.split_indices[split_name].append(len(self.samples))
                    self.samples.append(sample)

        print(
            f"[Info] {protocol_name} samples built: "
            f"users={used_users}, train={len(self.split_indices['train'])}, "
            f"val={len(self.split_indices['val'])}, test={len(self.split_indices['test'])}, "
            f"total={len(self.samples)}"
        )

    def _sample_excluding_items(
        self,
        exclude_items: set,
        rng: 'np.random.Generator',
        num_neg: int = 99,
    ):
        if len(exclude_items) >= max(1, self.num_items - 1):
            exclude_items = {0}
        negatives = []
        seen = set()
        while len(negatives) < num_neg:
            draw_size = max((num_neg - len(negatives)) * 3, 128)
            for neg in rng.integers(low=1, high=self.num_items, size=draw_size).tolist():
                neg = int(neg)
                if neg not in exclude_items and neg not in seen:
                    negatives.append(neg)
                    seen.add(neg)
                if len(negatives) == num_neg:
                    break
        return torch.tensor(negatives, dtype=torch.long)

    def _sample_popular_excluding_items(
        self,
        exclude_items: set,
        rng: 'np.random.Generator',
        popular_items: List[int],
        num_neg: int = 99,
        pool_size: int = 1500,
    ):
        candidate_pool = [
            int(item)
            for item in popular_items[:max(num_neg, pool_size)]
            if int(item) not in exclude_items and int(item) > 0
        ]
        if len(candidate_pool) < num_neg:
            seen = set(candidate_pool)
            for item in popular_items:
                item = int(item)
                if item > 0 and item not in exclude_items and item not in seen:
                    candidate_pool.append(item)
                    seen.add(item)
                    if len(candidate_pool) >= num_neg:
                        break
        if len(candidate_pool) < num_neg:
            return self._sample_excluding_items(exclude_items, rng, num_neg=num_neg)

        return torch.tensor(rng.choice(candidate_pool, size=num_neg, replace=False), dtype=torch.long)

    def _build_hem3bsr_loo_samples(
        self,
        data: 'pd.DataFrame',
        negative_strategy: str = 'uniform',
        skip_seen_eval_targets: bool = True,
    ):
        """Build samples following the referenced HEM3BSR experiment code.

        The last target behavior is test, the previous target behavior is
        validation, earlier target behaviors are training samples. Validation
        and test samples whose target item already appears in the current
        history are skipped. Fixed negatives exclude all items interacted by
        the user. The popneg variant samples fixed negatives from popular
        non-interacted items to avoid an overly easy random candidate set.
        """
        rng = np.random.default_rng(2025)
        self.samples = []
        self.split_indices = {'train': [], 'val': [], 'test': []}
        total_users = data['user_id'].nunique()
        item_popularity = data['item_id'].astype(int).value_counts()
        popular_items = item_popularity.index.astype(int).tolist()
        print(
            f"[Info] Building HEM3BSR leave-one-out samples for {total_users} users "
            f"(negative_strategy={negative_strategy}, skip_seen_eval_targets={skip_seen_eval_targets})..."
        )

        for user_count, (_, df_u) in enumerate(data.groupby('user_id'), start=1):
            if user_count % 5000 == 0:
                print(f"[Info] Processed {user_count}/{total_users} users, samples={len(self.samples)}")
            df_u = df_u.sort_values('timestamp') if 'timestamp' in df_u.columns else df_u
            rows = list(df_u.itertuples(index=False))
            user_all_items = {int(getattr(row, 'item_id')) for row in rows}
            target_positions = [
                idx for idx, row in enumerate(rows)
                if getattr(row, 'behavior', 'favor') == 'favor'
            ]
            if len(target_positions) < 3:
                continue

            split_for_pos = {target_positions[-1]: 'test', target_positions[-2]: 'val'}
            for pos in target_positions[:-2]:
                split_for_pos[pos] = 'train'

            click_hist: List[int] = []
            favor_hist: List[int] = []
            item_hist: List[int] = []
            behavior_hist: List[int] = []

            for idx, row in enumerate(rows):
                item = int(getattr(row, 'item_id'))
                beh = getattr(row, 'behavior', 'favor')

                if idx in split_for_pos and (click_hist or favor_hist):
                    split_name = split_for_pos[idx]
                    if skip_seen_eval_targets and split_name in {'val', 'test'} and (item in click_hist or item in favor_hist):
                        item_hist.append(item)
                        behavior_hist.append(1 if beh == 'favor' else 0)
                        if beh == 'favor':
                            favor_hist.append(item)
                        else:
                            click_hist.append(item)
                        continue

                    sample = {
                        'item_sequence': torch.tensor(item_hist[-self.seq_len:], dtype=torch.long),
                        'behavior_sequence': torch.tensor(behavior_hist[-self.seq_len:], dtype=torch.long),
                        'labels': torch.tensor(item, dtype=torch.long),
                        'click_id_seq': torch.tensor(click_hist[-self.seq_len:], dtype=torch.long),
                        'click_img_seq': torch.tensor(click_hist[-self.seq_len:], dtype=torch.long),
                        'click_txt_seq': torch.tensor(click_hist[-self.seq_len:], dtype=torch.long),
                        'favor_id_seq': torch.tensor(favor_hist[-self.seq_len:], dtype=torch.long),
                        'favor_img_seq': torch.tensor(favor_hist[-self.seq_len:], dtype=torch.long),
                        'favor_txt_seq': torch.tensor(favor_hist[-self.seq_len:], dtype=torch.long),
                    }
                    if split_name in {'val', 'test'}:
                        if negative_strategy == 'popularity':
                            sample['negatives'] = self._sample_popular_excluding_items(
                                user_all_items,
                                rng,
                                popular_items,
                                num_neg=99,
                                pool_size=self.hard_neg_pool_size,
                            )
                        else:
                            sample['negatives'] = self._sample_excluding_items(user_all_items, rng, num_neg=99)

                    self.samples.append(sample)
                    self.split_indices[split_name].append(len(self.samples) - 1)

                item_hist.append(item)
                behavior_hist.append(1 if beh == 'favor' else 0)
                if beh == 'favor':
                    favor_hist.append(item)
                else:
                    click_hist.append(item)

        print(
            "[Info] HEM3BSR LOO samples built: "
            f"train={len(self.split_indices['train'])}, "
            f"val={len(self.split_indices['val'])}, "
            f"test={len(self.split_indices['test'])}, "
            f"total={len(self.samples)}"
        )

    def _prepare(self):
        # 因高频数据过滤：根据过滤参数生成缓存文件名，避免不同参数使用相同缓存
        protocol_part = ''
        if self.data_protocol in {
            'loo',
            'loo_fixedneg',
            'strict_loo',
            'strict_loo_fixedneg',
            'mmsr_loo',
            'mmsr_loo_fixedneg',
            'mmsr_newitem',
            'mmsr_newitem_fixedneg',
            'hem3bsr_loo_fixedneg',
            'hem3bsr_loo_popneg',
            'taobao_loo_popneg',
        }:
            protocol_part = f"_{self.data_protocol}"
        hard_neg_part = f"_pool{self.hard_neg_pool_size}" if self.data_protocol in {'hem3bsr_loo_popneg', 'taobao_loo_popneg'} else ''
        target_part = "_target" + "-".join(sorted(self.target_behavior_set))
        auxiliary_part = "_aux" + "-".join(sorted(self.auxiliary_behavior_set)) if self.auxiliary_behavior_set else ''
        source_part = f"_src{self.csv_source}"
        cache_suffix = f"_{self.item_id_source}{protocol_part}{hard_neg_part}{target_part}{auxiliary_part}{source_part}_seq{self.seq_len}_item{self.min_item_freq}_user{self.min_user_freq}"
        cache_file = os.path.join(self.data_root, f'processed_data{cache_suffix}.pkl')
        if self.data_protocol == 'random' and self.csv_source == 'auto' and not os.path.exists(cache_file):
            legacy_suffix = f"_{self.item_id_source}_seq{self.seq_len}_item{self.min_item_freq}_user{self.min_user_freq}"
            legacy_cache_file = os.path.join(self.data_root, f'processed_data{legacy_suffix}.pkl')
            if os.path.exists(legacy_cache_file):
                cache_file = legacy_cache_file
        if os.path.exists(cache_file):
            print(f"[Info] 发现缓存文件，直接加载...")
            import pickle
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.samples = cached_data['samples']
                self.num_items = cached_data['num_items']
                self.item_id_source = cached_data.get('item_id_source', self.item_id_source)
                self.hard_neg_pool_size = cached_data.get('hard_neg_pool_size', self.hard_neg_pool_size)
                self.target_behavior_set = set(cached_data.get('target_behavior_set', self.target_behavior_set))
                self.auxiliary_behavior_set = set(cached_data.get('auxiliary_behavior_set', self.auxiliary_behavior_set))
                self.csv_source = cached_data.get('csv_source', self.csv_source)
                self.item2idx = cached_data.get('item2idx', {})
                self.split_indices = cached_data.get('split_indices', {})
                print(f"[Info] 从缓存加载 {len(self.samples)} 个样本，num_items={self.num_items}")
                return
        
        print(f"[Info] 首次处理数据，将创建缓存...")
        data = self._load_merge()
        data = self._split_behaviors(data)
        data = self._filter_high_freq_data(data)  # 因高频数据过滤：在map_ids之前过滤，避免映射后重新映射
        data = self._time_sort(data)
        data = self._map_ids(data)
        if self.data_protocol in {'hem3bsr_loo_fixedneg', 'hem3bsr_loo_popneg', 'taobao_loo_popneg'}:
            self._build_hem3bsr_loo_samples(
                data,
                negative_strategy='popularity' if self.data_protocol in {'hem3bsr_loo_popneg', 'taobao_loo_popneg'} else 'uniform',
                skip_seen_eval_targets=self.data_protocol != 'taobao_loo_popneg',
            )
        elif self.data_protocol in {'mmsr_loo', 'mmsr_loo_fixedneg', 'mmsr_newitem', 'mmsr_newitem_fixedneg'}:
            self._build_mmsr_loo_samples(
                data,
                fixed_negatives=self.data_protocol in {'mmsr_loo_fixedneg', 'mmsr_newitem_fixedneg'},
                new_item_only=self.data_protocol in {'mmsr_newitem', 'mmsr_newitem_fixedneg'},
            )
        elif self.data_protocol in {'strict_loo', 'strict_loo_fixedneg'}:
            self._build_strict_loo_samples(data, fixed_negatives=(self.data_protocol == 'strict_loo_fixedneg'))
        else:
            self._build_samples(data)
        if not self.samples:
            raise RuntimeError("KuaiLive 数据解析后无样本，请检查CSV内容与行为字段映射。")
        
        # 保存缓存
        import pickle
        cache_data = {
            'samples': self.samples,
            'num_items': self.num_items,
            'item_id_source': self.item_id_source,
            'hard_neg_pool_size': self.hard_neg_pool_size,
            'target_behavior_set': sorted(self.target_behavior_set),
            'auxiliary_behavior_set': sorted(self.auxiliary_behavior_set),
            'csv_source': self.csv_source,
            'item2idx': self.item2idx,
            'split_indices': self.split_indices,
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[Info] 数据已缓存到 {cache_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_dataset(
    data_root: str,
    num_items: Optional[int],
    seq_len: int,
    mode: str = 'multi_behavior',
    min_item_freq: int = 5,
    min_user_freq: int = 5,
    data_protocol: str = 'mmsr_loo_fixedneg',
    item_id_source: str = 'auto',
    hard_neg_pool_size: int = 1500,
    target_behaviors: Optional[str] = None,
    auxiliary_behaviors: Optional[str] = None,
    csv_source: str = 'auto',
) -> Dataset:
    # 因接入真实数据：将原先固定回退为合成数据的逻辑注释，改为优先解析 KuaiLive
    # kuailive_exists = os.path.exists(data_root)
    # if not kuailive_exists:
    #     print(f"[Info] KuaiLive data not found at {data_root}. Using synthetic dataset.")
    #     return SyntheticMultiBehaviorDataset(num_items=num_items, seq_len=seq_len, size=1024)
    # print(f"[Info] KuaiLive data found at {data_root}, but parser not implemented yet. Using synthetic dataset.")
    # return SyntheticMultiBehaviorDataset(num_items=num_items, seq_len=seq_len, size=2048)

    if os.path.exists(data_root) and pd is not None:
        try:
            ds = KuaiLiveDataset(
                data_root=data_root,
                seq_len=seq_len,
                min_item_freq=min_item_freq,
                min_user_freq=min_user_freq,
                data_protocol=data_protocol,
                item_id_source=item_id_source,
                hard_neg_pool_size=hard_neg_pool_size,
                target_behaviors=target_behaviors,
                auxiliary_behaviors=auxiliary_behaviors,
                csv_source=csv_source,
            )
            print(f"[Info] Loaded KuaiLive dataset with {len(ds)} samples, num_items={ds.num_items}")
            return ds
        except Exception as e:
            print(f"[Warn] KuaiLive 解析失败，回退为合成数据。原因: {e}")
    else:
        if not os.path.exists(data_root):
            print(f"[Info] KuaiLive data not found at {data_root}.")
        if pd is None:
            print("[Info] pandas/numpy 不可用，无法解析 KuaiLive。")
    # 回退为合成数据
    fallback_num_items = num_items if num_items is not None else 1000
    print(f"[Info] Using synthetic dataset. num_items={fallback_num_items}")
    return SyntheticMultiBehaviorDataset(num_items=fallback_num_items, seq_len=seq_len, size=1024)
