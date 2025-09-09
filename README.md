# DPSC: Dual-Path Semantic Consolidation

[ICASSP 2026] **DISCRIMINATIVE TEXT INTEGRATION AND SEMANTIC DIFFERENCE ENHANCEMENT
FOR FEW-SHOT CLASS-INCREMENTAL LEARNING**
---

## ✨ Abstract

Few-shot class-incremental learning (FSCIL) aims to contin-
ually incorporate novel categories with limited samples while re-
taining recognition of previously learned classes. Existing methods
typically construct a separable and stable embedding space dur-
ing the base stage to enhance base classes discriminability and
reserve space for novel classes, and fine-tune only the classifier in
incremental sessions. Although effective in alleviating catastrophic
forgetting, this strategy relies on scarce samples without sufficient
semantic constraints during the incremental stage, which easily leads
to overfitting and unstable representations. We argue that a robust
FSCIL approach should jointly address forgetting and overfitting
as complementary challenges. To this end, we propose Dual-Path
Semantic Consolidation (DPSC). For mitigating forgetting, we
design Discriminative Text-guided Visual Alignment (DTA) and
Semantic Differential Boundary Augmentation (SDA): the former
leverages the structured semantic space of CLIP and discriminative
textual descriptions from LLMs to guide visual embeddings toward
greater separability, while the latter constructs boundary negatives
from class-level textual differences to explicitly enlarge margins
between neighboring categories. For alleviating overfitting, we in-
troduce Language-enhanced Prototype Calibration (LPC), which
incorporates textual priors to decompose and recompose visual fea-
tures, thereby stabilizing novel-class prototypes during incremental
updates and enhancing generalization. Extensive experiments val-
idate DPSC’s state-of-the-art performance.

DPSC achieves **state-of-the-art performance** on:
- CIFAR100
- CUB200
- miniImageNet

---

## 📦 Requirements

- Python 3.9
- PyTorch ≥ 2.1
- tqdm

## Pipline
The whole learning pipline of our model:
<img width="1050" height="502" alt="截屏2025-09-09 14 47 51" src="https://github.com/user-attachments/assets/85f414b0-f42c-4303-935f-9c4b1ed01943" />

## ⌚️ Results

<img width="1103" height="669" alt="截屏2025-09-09 14 49 54" src="https://github.com/user-attachments/assets/5b7efe83-7b61-469a-bea9-e06eb3341f5f" />

## 🔥 Training scripts
  - Cifar100
    ```bash
    python train_noFan.py -project dpsc -dataset cifar100 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -moco_k 8192 -moco_t 0.07 -moco_m 0.995 -size_crops 32 18 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 4 -alpha 0.2 -beta 0.8 -constrained_cropping -use_text
  - CUB200
    ```bash
    python train_noFan.py -project dpsc -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.002 -lr_new 0.000005 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -moco_k 8192  -moco_t 0.07 -moco_m 0.999 -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 4 -constrained_cropping -alpha 0.2 -beta 0.8 -use_text
  - miniImagenet
     ```bash
    python train.py -project savc -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 40 70 100  -gpu '0' -temperature 16 -moco_dim 128 -moco_k 8192 -moco_t 0.07 -moco_m 0.999 -size_crops 84 50 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 4 -constrained_cropping -alpha 0.2 -beta 0.8 -use_text
     
Remember to change YOURDATAROOT into your own data root. If you want to use incremental finetuning, set -incft.

## 😄 Contact
If there are any questions, please feel free to contact with the author: Shilong Wang (wangshilong@nynu.edn.cn).
