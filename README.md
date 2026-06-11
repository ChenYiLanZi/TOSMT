 # TOSMT

TOSMT (Target-Oriented Stable Modal Transfer) is a multi-modal multi-behavior sequential recommendation model adapted from the original HEM3BSR/M3BSR codebase.

The current version keeps the original data loading, candidate sampling, training loop, and HR/NDCG evaluation flow, while replacing the original diffusion denoising and MEIE modules with:

- base multi-behavior sequence encoding
- sample-wise behavior-modal reliability estimation
- stable/noisy modal decomposition
- reliability-controlled stable modal transfer
- orthogonality, alignment, and sparsity auxiliary losses

## Project Structure

```text
TOSMT-main/
├── data_loader.py
├── train_hem3bsr.py
└── models/
    ├── hem3bsr_model.py
    └── modules.py
```

`train_hem3bsr.py` and class name `HEM3BSR` are kept for compatibility with the original scripts.

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- tqdm

## Quick Run

The KuaiLive dataset should be placed at:

```text
data/KuaiLive/
├── comment.csv
├── gift.csv
├── like.csv
└── title_embeddings.npy
```

```bash
python train_hem3bsr.py \
  --data_root ./data/KuaiLive \
  --text_embeddings_path ./data/KuaiLive/title_embeddings.npy \
  --epochs 3 \
  --batch_size 32 \
  --seq_len 20 \
  --d_model 128 \
  --lr 1e-4 \
  --num_neg 99 \
  --max_steps 100 \
  --max_eval_steps 50
```

If `--image_embeddings_path` is not provided, the model uses mock image embeddings.

For this KuaiLive copy, `streamer_id` is used as the item id because `live_id` exceeds the row range of `title_embeddings.npy`. The training script automatically remaps `title_embeddings.npy` to the contiguous item ids built by `data_loader.py`.

## Main TOSMT Parameters

- `--image_embeddings_path`: optional `.npy` image embedding path
- `--lambda_t`: stable modal transfer strength
- `--lambda_orth`: orthogonality loss weight
- `--lambda_align`: alignment loss weight
- `--lambda_sparse`: reliability sparsity loss weight
- `--ablation`: `none`, `ours`, `base_only`, `naive_fusion`, `no_reliability`, `no_decompose`, `no_align`, or `no_sparse`

## Notes

The first-stage implementation still uses the original random split and candidate-sampling evaluation. For the final paper version, update the data protocol to leave-one-out evaluation with 99 negative samples per target behavior.
