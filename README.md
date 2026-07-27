# <p align=center>`Skating-Mixer: Multimodal MLP for Scoring Figure Skating`</p><!-- omit in toc -->
The implementation of [AAAI2023 paper](https://arxiv.org/pdf/2203.03990.pdf).

# Introduction
Figure skating scoring is a challenging task because it requires judging players’ technical moves as well as coordination with the background music. Prior learning-based work cannot solve it well for two reasons: 1) each move in figure skating changes quickly, hence simply applying traditional frame sampling will lose a lot of valuable information, especially in a 3-5 minutes lasting video, so an extremely long-range representation learning is necessary; 2) prior methods rarely considered the critical audio-visual relationship in their models. Thus, we introduce a multimodal MLP architecture, named Skating-Mixer. It extends the MLP-Mixer-based framework into a multimodal fashion and effectively learns long-term representations through our designed memory recurrent unit (MRU). Aside from the model, we also collected a high-quality audio-visual FS1000 dataset, which contains over 1000 videos on 8 types of programs with 7 different rating metrics, overtaking other datasets in both quantity and diversity. Experiments show the proposed method outperforms SOTAs over all major metrics on the public Fis-V and our FS1000 dataset. In addition, we include an analysis applying our method to recent competitions that occurred in Beijing 2022 Winter Olympic Games, proving our method has strong robustness.

# Dataset
The proposed Timesformer, AST, C3D and VGGish feature of the proposed dataset can be found [here](https://pan.baidu.com/s/1SGbvK6vDGR7ZP0PxakUO7g?pwd=9tma).
Also, if you need the raw videos, please require jingfeixia708@gmail.com.

# Citation
```
@article{xia2022skating,
  title={Skating-Mixer: Multimodal MLP for Scoring Figure Skating},
  author={Xia, Jingfei and Zhuge, Mingchen and Geng, Tiantian and Fan, Shun and Wei, Yuantai and He, Zhenyu and Zheng, Feng},
  journal={arXiv preprint arXiv:2203.03990},
  year={2022}
}
```

# Simplified Action-RAG experiment

The `rag` branch implements a controlled retrieval-augmented TES residual:

```text
TimeSformer + AST + bidirectional MRU
    -> masked global pooling -> TES_dynamic

FS1000 DINO query
    -> fixed cosine Top-K retrieval from a FineFS action corpus
    -> query/reference pair encoder with GOE, BV and panel-score evidence
    -> evidence-only video aggregation -> Delta_TES_RAG

TES_final = TES_dynamic + Delta_TES_RAG
```

The correction head never receives `dynamic_global` directly. A correction is
forced to zero when no retrieved candidate has valid action-score evidence. The
corpus tensors are frozen and are not stored in each checkpoint.

## Data and leakage rules

- The existing `train_fs800.txt` and `val_fs800.txt` files are unchanged.
- FineFS is the labelled external reference set.
- FineFS videos that uniquely match an FS1000 validation score signature are
  removed from the corpus entirely.
- For an FS1000 training query that uniquely matches a FineFS video, all
  prototypes from that FineFS video are excluded during retrieval.
- Non-unique score signatures are never guessed.
- Existing DINO feature arrays are not overwritten. New FineFS features use
  `static_dinov2_cls_patch_mean_rag_cache`, and FS1000 receives only new
  `.times.npy` sidecars.

## Environments

No package installation is required. Preprocessing uses
`/home/v100/anaconda3/envs/skating-dinov2`; model training and evaluation use
`/home/v100/anaconda3/envs/skating-action`.

## Commands

All commands are collected in `scripts/run_action_rag.sh`:

```bash
bash scripts/run_action_rag.sh test
bash scripts/run_action_rag.sh preprocess-all
bash scripts/run_action_rag.sh train-dynamic

DYNAMIC_CHECKPOINT=/absolute/path/to/dynamic_best.pth \
  bash scripts/run_action_rag.sh train-rag

CHECKPOINT=/absolute/path/to/rag_best.pth \
  bash scripts/run_action_rag.sh evaluate

CHECKPOINT=/absolute/path/to/rag_best.pth \
  bash scripts/run_action_rag.sh evaluate-dynamic
```

Best checkpoints are selected by validation Spearman and named with epoch,
validation loss and Spearman. The script-generated run name already contains
the start time, for example
`dynamic_seed2026_20260721_120000_best_epoch012_loss64.7400_spear0.8710.pth`.

The last command disables the RAG residual in the same checkpoint, providing a
matched dynamic-only control for the final evaluation.

## FineFS semantic supervision stage

Before TES residual training, FineFS can now directly supervise action
retrieval, citation reranking, action/background recognition and GOE. The split
is video-disjoint and immutable by default. Only FineFS-train videos form the
reference bank; train queries exclude their own video, while validation and
test videos never enter the bank.

```bash
# Create the deterministic 70/15/15 video split once.
bash scripts/run_action_rag.sh semantic-split

# Short real-corpus check before a full run.
TRAIN_GPU=0 bash scripts/run_action_rag.sh semantic-smoke

# Full semantic training.
TRAIN_GPU=0 bash scripts/run_action_rag.sh train-semantic

# Validation and held-out FineFS test evaluation.
SEMANTIC_CHECKPOINT=/absolute/path/to/semantic_best.pth TRAIN_GPU=0 \
  bash scripts/run_action_rag.sh semantic-val
SEMANTIC_CHECKPOINT=/absolute/path/to/semantic_best.pth TRAIN_GPU=0 \
  bash scripts/run_action_rag.sh semantic-test
```

The semantic checkpoint is independent of previous RAG checkpoints. It uses a
2048 -> 512 -> 256 DINO query encoder and stores the full retrieval/citation/GOE
model for a newly trained TES stage. FineFS test labels are used only for final
semantic metrics.

`preprocess-all` is intentionally separate from training because FineFS DINO
feature extraction covers 1167 long videos and can take several hours. The
step is resumable and skips completed feature/time pairs.
