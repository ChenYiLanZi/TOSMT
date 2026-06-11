#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import pickle
from typing import Dict, Iterable, List

import numpy as np

from data_loader import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose candidate-set difficulty with a popularity baseline."
    )
    parser.add_argument("--data_root", type=str, default=os.path.join("data", "KuaiLive"))
    parser.add_argument("--item_id_source", type=str, default="streamer_id", choices=["auto", "item_id", "streamer_id", "live_id"])
    parser.add_argument("--data_protocol", type=str, default="hem3bsr_loo_popneg")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--min_item_freq", type=int, default=10)
    parser.add_argument("--min_user_freq", type=int, default=10)
    parser.add_argument("--num_neg", type=int, default=99)
    parser.add_argument("--pool_sizes", type=str, default="1000,1500,2000")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--target_behaviors", type=str, default=None)
    parser.add_argument("--auxiliary_behaviors", type=str, default=None)
    parser.add_argument("--csv_source", type=str, default="auto", choices=["auto", "behavior_files"])
    parser.add_argument("--target_hr10", type=float, default=0.2728)
    parser.add_argument("--target_ndcg10", type=float, default=0.1709)
    parser.add_argument("--target_hr20", type=float, default=0.3311)
    parser.add_argument("--target_ndcg20", type=float, default=0.1714)
    parser.add_argument("--cache_path", type=str, default=None, help="Evaluate one existing processed_data*.pkl cache directly.")
    return parser.parse_args()


def tensor_to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def tensor_to_list(value) -> List[int]:
    if hasattr(value, "tolist"):
        return [int(v) for v in value.tolist()]
    return [int(v) for v in value]


def load_cache(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def dataset_to_cache(dataset) -> Dict:
    return {
        "samples": dataset.samples,
        "split_indices": dataset.split_indices,
        "num_items": dataset.num_items,
        "item_id_source": getattr(dataset, "item_id_source", None),
        "hard_neg_pool_size": getattr(dataset, "hard_neg_pool_size", None),
    }


def compute_train_popularity(cache: Dict) -> np.ndarray:
    pop = np.zeros(int(cache["num_items"]), dtype=np.float64)
    samples = cache["samples"]
    for idx in cache["split_indices"]["train"]:
        pop[tensor_to_int(samples[idx]["labels"])] += 1.0
    return pop


def popularity_metrics(cache: Dict, split: str, num_neg: int, topks: Iterable[int] = (10, 20)) -> Dict[str, float]:
    samples = cache["samples"]
    split_indices = cache["split_indices"][split]
    pop = compute_train_popularity(cache)
    topks = tuple(topks)
    hits = {k: 0 for k in topks}
    ndcgs = {k: 0.0 for k in topks}

    for sample_idx in split_indices:
        sample = samples[sample_idx]
        pos = tensor_to_int(sample["labels"])
        if "negatives" not in sample:
            raise ValueError("Cache samples do not contain fixed negatives; use a *_fixedneg or *_popneg protocol.")
        negatives = [n for n in tensor_to_list(sample["negatives"]) if n > 0 and n != pos][:num_neg]
        candidates = [pos] + negatives
        scores = [pop[c] for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        rank = order.index(0) + 1

        for k in topks:
            if rank <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / math.log2(rank + 1)

    total = max(1, len(split_indices))
    result = {}
    for k in topks:
        result[f"HR@{k}"] = hits[k] / total
        result[f"NDCG@{k}"] = ndcgs[k] / total
    return result


def get_targets(args) -> Dict[str, float]:
    return {
        "HR@10": args.target_hr10,
        "NDCG@10": args.target_ndcg10,
        "HR@20": args.target_hr20,
        "NDCG@20": args.target_ndcg20,
    }


def target_error(metrics: Dict[str, float], targets: Dict[str, float]) -> float:
    return sum(abs(metrics[name] - target) for name, target in targets.items())


def print_metrics(label: str, split: str, metrics: Dict[str, float], targets: Dict[str, float]):
    values = " ".join(
        f"{name}={metrics[name]:.4f}(diff={metrics[name] - targets[name]:+.4f})"
        for name in ("HR@10", "NDCG@10", "HR@20", "NDCG@20")
    )
    print(f"{label:>12} {split:>4} {values} total_abs_diff={target_error(metrics, targets):.4f}")


def build_or_load_cache(args, pool_size: int) -> Dict:
    dataset = build_dataset(
        data_root=args.data_root,
        num_items=None,
        seq_len=args.seq_len,
        min_item_freq=args.min_item_freq,
        min_user_freq=args.min_user_freq,
        data_protocol=args.data_protocol,
        item_id_source=args.item_id_source,
        hard_neg_pool_size=pool_size,
        target_behaviors=args.target_behaviors,
        auxiliary_behaviors=args.auxiliary_behaviors,
        csv_source=args.csv_source,
    )
    return dataset_to_cache(dataset)


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    targets = get_targets(args)

    if args.cache_path:
        cache = load_cache(args.cache_path)
        label = f"pool{cache.get('hard_neg_pool_size', 'cache')}"
        print(f"cache={args.cache_path}")
        print(f"num_items={cache.get('num_items')} splits={ {k: len(v) for k, v in cache.get('split_indices', {}).items()} }")
        for split in splits:
            print_metrics(label, split, popularity_metrics(cache, split, args.num_neg), targets)
        return

    pool_sizes = [int(v.strip()) for v in args.pool_sizes.split(",") if v.strip()]
    best = None
    for pool_size in pool_sizes:
        cache = build_or_load_cache(args, pool_size)
        label = f"pool{pool_size}"
        print(f"\n{label}: num_items={cache.get('num_items')} splits={ {k: len(v) for k, v in cache.get('split_indices', {}).items()} }")
        for split in splits:
            metrics = popularity_metrics(cache, split, args.num_neg)
            print_metrics(label, split, metrics, targets)
            if split == "test":
                score = target_error(metrics, targets)
                if best is None or score < best[0]:
                    best = (score, pool_size, metrics)

    if best is not None:
        _, pool_size, metrics = best
        print("\nBest test pool_size by total absolute diff:")
        print_metrics(f"pool{pool_size}", "test", metrics, targets)


if __name__ == "__main__":
    main()
