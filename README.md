# DSC: Discriminative Semantic Consolidation

[ICASSP 2026] **TAILORED TEXT INTEGRATION AND SEMANTIC DIFFERENCE ENHANCEMENT
FOR FEW-SHOT CLASS-INCREMENTAL LEARNING**
---

## ✨ Abstract

Few-shot class-incremental learning (FSCIL) aims to continually incorporate novel classes with limited samples without forgetting previously learned ones. Existing methods typically construct a separable embedding space during the base session to enhance base-class discriminability and reserve space for novel classes. Subsequently, they fine-tune the classifier in incremental sessions. While effective in alleviating forgetting, these methods rely on scarce samples without sufficient semantic constraints, leading to overfitting and unstable representations. We argue that a robust FSCIL framework should jointly address the two challenges of forgetting and overfitting. To this end, we propose Discriminative Semantic Consolidation (DSC), which leverages tailored textual descriptions as discriminative semantic priors to address both challenges. For mitigating forgetting, we design Tailored Text-guided Visual Alignment (TTA) and Semantic Differential Boundary Augmentation (SDA). TTA employs the structured semantic space of CLIP and tailored textual descriptions from LLMs to guide visual embeddings toward greater separability, while SDA constructs boundary negatives from class-level textual differential to enlarge margins between neighboring categories. For alleviating overfitting, we introduce Language-enhanced Prototype Calibration (LPC), which employs discriminative semantic priors to decompose and recompose visual features, thereby stabilizing prototype updates and improving generalization. Extensive experiments demonstrate that DSC achieves state-of-the-art performance. 

DSC achieves **state-of-the-art performance** on:
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
<img width="882" height="481" alt="截屏2025-09-16 20 15 18" src="https://github.com/user-attachments/assets/2354c1e2-6fb4-44f8-a436-5188e7a07519" />


## ⌚️ Results

<img width="877" height="534" alt="截屏2025-09-16 20 14 20" src="https://github.com/user-attachments/assets/eaba4071-a076-467f-ae22-05d225a00b66" />


## 🔥 Training scripts
  - Cifar100
    ```bash
    python train_noFan.py -project dsc -dataset cifar100 -base_mode 'ft_cos' -new_mode 'avg_cos' -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 10 -schedule Cosine -gpu 0 -temperature 16 -moco_dim 32 -moco_k 8192 -moco_t 0.07 -moco_m 0.995 -size_crops 32 18 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -num_crops 2 4 -alpha 0.2 -beta 0.8 -constrained_cropping -use_text
  - CUB200
    ```bash
    python train_noFan.py -project dsc -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.0001 -lr_new 0.00001 -decay 0.0005 -epochs_base 30 -schedule Milestone -milestones 60 80 100  -gpu '0' -temperature 16 -moco_dim 128 -moco_k 8192  -moco_t 0.07 -moco_m 0.999 -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 4 -constrained_cropping -alpha 0.2 -beta 0.8 -use_text
  - miniImagenet
     ```bash
    python train.py -project dsc -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 10 -schedule Milestone -milestones 40 70 100  -gpu '0' -temperature 16 -moco_dim 128 -moco_k 8192 -moco_t 0.07 -moco_m 0.999 -size_crops 84 50 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -num_crops 2 4 -constrained_cropping -alpha 0.2 -beta 0.8 -use_text
     
Remember to change YOURDATAROOT into your own data root. If you want to use incremental finetuning, set -incft.

## 😄 Contact
If there are any questions, please feel free to contact with the author: Shilong Wang (wangshilong@nynu.edn.cn).
