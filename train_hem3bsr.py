#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TOSMT模型训练脚本
目标导向稳定模态转移的多模态多行为序列推荐模型
"""

import os
import argparse
import json
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data_loader import build_dataset
from models.hem3bsr_model import HEM3BSR


def parse_args():
    parser = argparse.ArgumentParser(description="TOSMT - Target-Oriented Stable Modal Transfer")
    parser.add_argument('--num_items', type=int, default=-1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)  # 因梯度爆炸问题降低学习率
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_root', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'KuaiLive'))
    parser.add_argument(
        '--data_protocol',
        type=str,
        default='mmsr_loo_fixedneg',
        choices=[
            'mmsr_loo_fixedneg',
            'mmsr_loo',
            'mmsr_newitem_fixedneg',
            'mmsr_newitem',
            'hem3bsr_loo_fixedneg',
            'hem3bsr_loo_popneg',
            'taobao_loo_popneg',
            'strict_loo_fixedneg',
            'strict_loo',
            'loo_fixedneg',
            'loo',
            'random',
        ],
        help='数据协议：默认使用MMSR/M3BSR实验设置的leave-one-out固定负样本缓存'
    )
    parser.add_argument('--num_neg', type=int, default=99, help='每个样本的负例个数')
    parser.add_argument('--hard_neg_pool_size', type=int, default=1500, help='hem3bsr_loo_popneg的热门负样本候选池大小')
    parser.add_argument('--max_steps', type=int, default=-1, help='每个epoch训练最大步数；-1表示全量')
    parser.add_argument('--max_eval_steps', type=int, default=-1, help='评测最大步数；-1表示全量')
    parser.add_argument('--eval_interval', type=int, default=1, help='每隔多少个epoch验证一次')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='验证NDCG@10连续多少次不提升后早停；<=0表示关闭')
    parser.add_argument('--selection_mode', type=str, default='best_ndcg10', choices=['best_ndcg10', 'target_distance'], help='checkpoint选择方式')
    parser.add_argument('--target_hr10', type=float, default=0.2728, help='target_distance模式的HR@10目标')
    parser.add_argument('--target_ndcg10', type=float, default=0.1709, help='target_distance模式的NDCG@10目标')
    parser.add_argument('--target_hr20', type=float, default=0.3311, help='target_distance模式的HR@20目标')
    parser.add_argument('--target_ndcg20', type=float, default=0.1714, help='target_distance模式的NDCG@20目标')
    parser.add_argument('--rank_diagnostics', action='store_true', help='评估时输出正例rank分布诊断')
    parser.add_argument('--metrics_path', type=str, default=None, help='可选：把最终测试指标写入JSON文件')
    parser.add_argument('--diffusion_timesteps', type=int, default=100, help='扩散时间步数，TOSMT中仅保留用于兼容旧命令')
    parser.add_argument('--text_embeddings_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'KuaiLive', 'title_embeddings.npy'), help='文本嵌入文件路径')
    parser.add_argument('--image_embeddings_path', type=str, default=None, help='图像嵌入文件路径；不提供时使用mock image embedding')
    parser.add_argument('--lambda_t', type=float, default=0.1, help='稳定模态迁移强度')
    parser.add_argument('--lambda_orth', type=float, default=0.01, help='稳定/噪声正交损失权重')
    parser.add_argument('--lambda_align', type=float, default=0.01, help='稳定成分与item embedding对齐损失权重')
    parser.add_argument('--lambda_sparse', type=float, default=0.001, help='可靠性稀疏损失权重')
    parser.add_argument(
        '--ablation',
        type=str,
        default='none',
        choices=[
            'none',
            'ours',
            'base_only',
            'direct_fusion',
            'naive_fusion',
            'no_reliability',
            'wo_rel',
            'global_reliability',
            'global_rel',
            'no_decompose',
            'wo_snd',
            'no_align',
            'no_sparse',
        ],
        help='消融实验模式'
    )
    # 因高频数据过滤：添加过滤参数
    parser.add_argument('--min_item_freq', type=int, default=10, help='item最小交互频率，低于此值的item将被过滤')
    parser.add_argument('--min_user_freq', type=int, default=10, help='user最小交互频率，低于此值的user将被过滤')
    parser.add_argument(
        '--item_id_source',
        type=str,
        default='auto',
        choices=['auto', 'item_id', 'streamer_id', 'live_id'],
        help='KuaiLive item列选择；参考HEM3BSR代码可尝试live_id'
    )
    parser.add_argument('--target_behaviors', type=str, default=None, help='Comma-separated target behaviors; use collect for Taobao.')
    parser.add_argument('--auxiliary_behaviors', type=str, default=None, help='Comma-separated auxiliary behaviors; use click for Taobao.')
    parser.add_argument('--csv_source', type=str, default='auto', choices=['auto', 'behavior_files'], help='CSV input source selection.')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    # batch is a list of dicts
    keys = batch[0].keys()
    result = {}
    
    def is_sequence_key(key):
        return key.endswith('_seq') or key.endswith('_sequence')

    # 找到所有序列数据的最大长度
    max_seq_len = 0
    for item in batch:
        for key in keys:
            if is_sequence_key(key) and len(item[key].shape) > 0:
                max_seq_len = max(max_seq_len, item[key].shape[0])
    
    for key in keys:
        tensors = [item[key] for item in batch]
        # 对于序列数据，需要padding到相同长度
        if is_sequence_key(key) and len(tensors[0].shape) > 0:
            # 使用全局最大长度进行padding
            padded_tensors = []
            for tensor in tensors:
                if tensor.shape[0] < max_seq_len:
                    # 使用0进行padding
                    if len(tensor.shape) == 1:
                        padding = torch.zeros(max_seq_len - tensor.shape[0], dtype=tensor.dtype)
                    else:
                        padding = torch.zeros(max_seq_len - tensor.shape[0], tensor.shape[1], dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding], dim=0)
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)
            result[key] = torch.stack(padded_tensors)
        else:
            result[key] = torch.stack(tensors)
    return result


def sample_negative_items(pos_item: int, history_tensors, num_items: int, num_neg: int, rng) -> list:
    exclude = {0, int(pos_item)}
    for tensor in history_tensors:
        for item in tensor.view(-1).tolist():
            item = int(item)
            if item > 0:
                exclude.add(item)

    if len(exclude) >= max(1, num_items - 1):
        exclude = {0, int(pos_item)}

    negs = []
    seen = set()
    while len(negs) < num_neg:
        draw_size = max((num_neg - len(negs)) * 2, 16)
        for n in rng.integers(low=1, high=num_items, size=draw_size).tolist():
            n = int(n)
            if n not in exclude and n not in seen:
                negs.append(n)
                seen.add(n)
                if len(negs) == num_neg:
                    break
    return negs


def build_candidate_batch(batch, device, num_items: int, num_neg: int, rng):
    labels_item = batch['labels']
    batch_size = labels_item.shape[0]
    negatives = batch.get('negatives')
    candidates = []

    for i in range(batch_size):
        pos = int(labels_item[i].item())
        if negatives is not None:
            neg = negatives[i].view(-1).tolist()
            neg = [int(n) for n in neg if int(n) != pos and int(n) > 0]
            if len(neg) < num_neg:
                extra = rng.integers(low=1, high=num_items, size=num_neg - len(neg)).tolist()
                neg.extend([n if n != pos else ((n % (num_items - 1)) + 1) for n in extra])
            neg = neg[:num_neg]
        else:
            neg = sample_negative_items(
                pos,
                [batch['click_id_seq'][i], batch['favor_id_seq'][i]],
                num_items,
                num_neg,
                rng,
            )
        candidates.append([pos] + neg)

    candidates = torch.tensor(candidates, dtype=torch.long, device=device)
    target_pos_in_cand = torch.zeros(batch_size, dtype=torch.long, device=device)
    return candidates, target_pos_in_cand


def evaluate_with_candidates(model, dataloader, device, num_items: int, num_neg: int, topk_list=[10, 20], max_eval_steps: int = -1, rank_diagnostics: bool = False):
    """候选采样评估方法（速度快）"""
    model.eval()
    rng = np.random.default_rng(123)
    hr_dict = {k: [] for k in topk_list}
    ndcg_dict = {k: [] for k in topk_list}
    rank_values = []
    total_samples = 0
    with torch.no_grad():
        step = 0
        for batch in dataloader:
            if max_eval_steps != -1 and step >= max_eval_steps:
                break
            step += 1
            B = batch['labels'].shape[0]
            total_samples += B
            click_id_seq = batch['click_id_seq'].to(device)
            click_img_seq = batch['click_img_seq'].to(device)
            click_txt_seq = batch['click_txt_seq'].to(device)
            favor_id_seq = batch['favor_id_seq'].to(device)
            favor_img_seq = batch['favor_img_seq'].to(device)
            favor_txt_seq = batch['favor_txt_seq'].to(device)
            candidates, target_pos_in_cand = build_candidate_batch(batch, device, num_items, num_neg, rng)
            
            loss, logits_cand = model(
                click_id_seq, click_img_seq, click_txt_seq,
                favor_id_seq, favor_img_seq, favor_txt_seq,
                target_pos_in_cand,
                candidate_indices=candidates,
                return_loss=True
            )
            scores = logits_cand.detach().cpu().numpy()
            for score in scores:
                ranked_all = np.argsort(-score)
                true_rank_arr = np.where(ranked_all == 0)[0]
                true_rank = int(true_rank_arr[0]) + 1 if len(true_rank_arr) > 0 else score.shape[0] + 1
                if rank_diagnostics:
                    rank_values.append(true_rank)
                for k in topk_list:
                    kk = min(k, score.shape[0])
                    if true_rank <= kk:
                        hr_dict[k].append(1)
                        ndcg_dict[k].append(1.0 / np.log2(true_rank + 1))
                    else:
                        hr_dict[k].append(0)
                        ndcg_dict[k].append(0)
                    continue
                    ranked_idx = np.argsort(-score)[:kk]
                    if 0 in ranked_idx:  # 正例在候选第0位
                        hr_dict[k].append(1)
                        true_rank = np.where(ranked_idx == 0)[0]
                        if len(true_rank) > 0:
                            ndcg_dict[k].append(1.0 / np.log2(true_rank[0] + 2))
                        else:
                            ndcg_dict[k].append(0)
                    else:
                        hr_dict[k].append(0)
                        ndcg_dict[k].append(0)
    res = {f'HR@{k}': np.mean(hr_dict[k]) for k in topk_list}
    res.update({f'NDCG@{k}': np.mean(ndcg_dict[k]) for k in topk_list})
    if rank_diagnostics and rank_values:
        ranks = np.asarray(rank_values)
        res['Rank@1-10'] = float(np.mean(ranks <= 10))
        res['Rank@11-20'] = float(np.mean((ranks > 10) & (ranks <= 20)))
        res['Rank@21+'] = float(np.mean(ranks > 20))
        res['MeanRank'] = float(np.mean(ranks))
    res['eval_samples'] = total_samples
    model.train()
    return res


def target_metric_distance(metric: dict, args) -> float:
    targets = {
        'HR@10': args.target_hr10,
        'NDCG@10': args.target_ndcg10,
        'HR@20': args.target_hr20,
        'NDCG@20': args.target_ndcg20,
    }
    return sum(abs(float(metric.get(name, 0.0)) - target) for name, target in targets.items())


def remap_external_embeddings(embedding_path: str, dataset, tag: str) -> str:
    if not embedding_path or not os.path.exists(embedding_path):
        return embedding_path

    item2idx = getattr(dataset, 'item2idx', None)
    num_items = getattr(dataset, 'num_items', None)
    item_id_source = getattr(dataset, 'item_id_source', 'item')
    if not item2idx or num_items is None:
        return embedding_path

    base, ext = os.path.splitext(embedding_path)
    remapped_path = f"{base}.{item_id_source}.n{num_items}.remapped{ext}"
    if f".n{num_items}.remapped" in embedding_path:
        print(f"[Info] Using already remapped {tag} embeddings: {embedding_path}")
        return embedding_path

    if os.path.exists(remapped_path):
        arr = np.load(remapped_path, mmap_mode='r')
        if arr.shape[0] == num_items:
            print(f"[Info] Using remapped {tag} embeddings: {remapped_path}, shape={arr.shape}")
            return remapped_path

    try:
        source = np.load(embedding_path, mmap_mode='r')
    except ValueError as exc:
        raise ValueError(
            f"Failed to load {tag} embeddings from {embedding_path}. "
            f"The file may be incomplete or corrupted. Re-upload it or pass an already remapped file."
        ) from exc
    remapped = np.zeros((num_items, source.shape[1]), dtype=np.float32)
    missing = 0
    for raw_id, mapped_id in item2idx.items():
        try:
            raw_idx = int(raw_id)
        except ValueError:
            missing += 1
            continue
        if 0 <= raw_idx < source.shape[0]:
            remapped[mapped_id] = source[raw_idx]
        else:
            missing += 1

    np.save(remapped_path, remapped)
    print(
        f"[Info] Saved remapped {tag} embeddings: {remapped_path}, "
        f"shape={remapped.shape}, missing_items={missing}"
    )
    return remapped_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用设备: {device}")
    print(f"[Info] 随机种子: {args.seed}")

    # 创建完整数据集
    full_dataset = build_dataset(
        data_root=args.data_root,
        num_items=(None if args.num_items == -1 else args.num_items),
        seq_len=args.seq_len,
        mode='multi_behavior',
        min_item_freq=args.min_item_freq,  # 因高频数据过滤：传递过滤参数
        min_user_freq=args.min_user_freq,  # 因高频数据过滤：传递过滤参数
        data_protocol=args.data_protocol,
        item_id_source=args.item_id_source,
        hard_neg_pool_size=args.hard_neg_pool_size,
        target_behaviors=args.target_behaviors,
        auxiliary_behaviors=args.auxiliary_behaviors,
        csv_source=args.csv_source,
    )
    
    total_samples = len(full_dataset)
    split_indices = getattr(full_dataset, 'split_indices', {})
    
    if all(name in split_indices for name in ('train', 'val', 'test')):
        train_indices = split_indices['train']
        val_indices = split_indices['val']
        test_indices = split_indices['test']
        print(f"[Info] 使用数据缓存中的split_indices划分: protocol={args.data_protocol}")
    else:
        # 旧缓存回退：简单的数据划分：70%训练，15%验证，15%测试
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_samples))
        print("[Info] 缓存中未发现split_indices，回退到70/15/15顺序划分")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"[Info] 数据划分: 训练集{len(train_indices)}样本, 验证集{len(val_indices)}样本, 测试集{len(test_indices)}样本")

    inferred_num_items = getattr(full_dataset, 'num_items', None)
    num_items_for_model = inferred_num_items if inferred_num_items is not None else (args.num_items if args.num_items != -1 else 1000)
    text_embeddings_path = remap_external_embeddings(args.text_embeddings_path, full_dataset, 'text')
    image_embeddings_path = remap_external_embeddings(args.image_embeddings_path, full_dataset, 'image')

    # 创建TOSMT模型；类名保留为HEM3BSR以兼容旧脚本导入
    model = HEM3BSR(
        num_items=num_items_for_model,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        diffusion_timesteps=args.diffusion_timesteps,
        text_embeddings_path=text_embeddings_path,
        image_embeddings_path=image_embeddings_path,
        lambda_t=args.lambda_t,
        lambda_orth=args.lambda_orth,
        lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse,
        ablation=args.ablation
    ).to(device)

    print(f"[Info] TOSMT模型参数数量: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 因梯度爆炸问题添加学习率调度器
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6  # 注释：因verbose参数错误而修改
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 创建随机数生成器（在训练循环外，避免重复创建）
    rng = np.random.default_rng()
    best_val_ndcg10 = float('-inf')
    best_selection_score = float('inf') if args.selection_mode == 'target_distance' else float('-inf')
    best_val_metrics = None
    best_epoch = 0
    best_state = None
    bad_eval_count = 0
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0
        
        # 创建进度条
        max_steps_display = args.max_steps if args.max_steps != -1 else len(train_dataloader)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", total=max_steps_display)
        
        step = 0
        for batch in pbar:
            if args.max_steps != -1 and step >= args.max_steps:
                break
            step += 1

            B = batch['labels'].shape[0]
            click_id_seq = batch['click_id_seq'].to(device)
            click_img_seq = batch['click_img_seq'].to(device)
            click_txt_seq = batch['click_txt_seq'].to(device)
            favor_id_seq = batch['favor_id_seq'].to(device)
            favor_img_seq = batch['favor_img_seq'].to(device)
            favor_txt_seq = batch['favor_txt_seq'].to(device)
            candidates, target_pos_in_cand = build_candidate_batch(
                batch, device, num_items_for_model, args.num_neg, rng
            )

            optimizer.zero_grad()
            loss, logits = model(
                click_id_seq, click_img_seq, click_txt_seq,
                favor_id_seq, favor_img_seq, favor_txt_seq,
                target_pos_in_cand,
                candidate_indices=candidates,
                return_loss=True
            )
            
            # 因梯度爆炸问题添加NaN检查和梯度裁剪
            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"[Warning] 检测到NaN loss，跳过此批次 (step {step})")
                continue
            
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条显示当前loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(1, num_batches)
        pbar.close()
        print(f"Epoch {epoch+1}/{args.epochs} - avg_loss: {avg_loss:.4f}")
        
        # 因梯度爆炸问题添加学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"[Info] 学习率从 {old_lr:.2e} 降低到 {new_lr:.2e}")
        
        should_eval = args.eval_interval > 0 and ((epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs)
        if should_eval:
            metric = evaluate_with_candidates(model, val_dataloader, device, num_items_for_model, args.num_neg, max_eval_steps=args.max_eval_steps, rank_diagnostics=args.rank_diagnostics)
            eval_samples = metric.pop('eval_samples', 0)
            metric_str = ', '.join([f'{k}: {v:.4f}' for k,v in metric.items()])
            print(f"[Eval-Candidate][epoch {epoch+1}] (samples={eval_samples}, candidates={args.num_neg+1}) {metric_str}")

            val_ndcg10 = metric.get('NDCG@10', float('-inf'))
            if args.selection_mode == 'target_distance':
                selection_score = target_metric_distance(metric, args)
                improved = selection_score < best_selection_score
                selection_label = f"target distance={selection_score:.4f}"
            else:
                selection_score = val_ndcg10
                improved = selection_score > best_selection_score
                selection_label = f"NDCG@10={selection_score:.4f}"

            if improved:
                best_selection_score = selection_score
                best_val_ndcg10 = val_ndcg10
                best_val_metrics = dict(metric)
                best_epoch = epoch + 1
                bad_eval_count = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[Info] New best validation {selection_label} at epoch {best_epoch}")
            else:
                bad_eval_count += 1
                print(f"[Info] Validation selection did not improve ({bad_eval_count}/{args.early_stop_patience})")

            if args.early_stop_patience > 0 and bad_eval_count >= args.early_stop_patience:
                print(f"[Info] Early stopping at epoch {epoch+1}; best epoch={best_epoch}, selection_mode={args.selection_mode}, best_score={best_selection_score:.4f}, best NDCG@10={best_val_ndcg10:.4f}")
                break
    
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"[Info] Loaded best validation model from epoch {best_epoch} (selection_mode={args.selection_mode}, score={best_selection_score:.4f}, NDCG@10={best_val_ndcg10:.4f})")

    # 训练结束后，在测试集上进行最终评估（全量测试集）
    print("\n=== 测试集评估 ===")
    test_metric = evaluate_with_candidates(model, test_dataloader, device, num_items_for_model, args.num_neg, max_eval_steps=args.max_eval_steps, rank_diagnostics=args.rank_diagnostics)
    test_eval_samples = test_metric.pop('eval_samples', 0)
    test_metric_str = ', '.join([f'{k}: {v:.4f}' for k,v in test_metric.items()])
    print(f"[Test-Candidate] (samples={test_eval_samples}, candidates={args.num_neg+1}) {test_metric_str}")
    if args.metrics_path:
        metrics_dir = os.path.dirname(args.metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        payload = {
            'best_epoch': best_epoch,
            'best_val_ndcg10': best_val_ndcg10,
            'best_selection_score': best_selection_score,
            'best_val_metrics': best_val_metrics,
            'test_eval_samples': test_eval_samples,
            'num_candidates': args.num_neg + 1,
            'args': vars(args),
            'test_metrics': test_metric,
        }
        with open(args.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Info] Saved metrics to {args.metrics_path}")


if __name__ == '__main__':
    main()
