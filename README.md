# FairMGA: Gradient Modulation Framework for Fair Multimodal Learning

An official implementation of **FairMGA** (Fairness-aware Multimodal Gradient Allocation). This framework mitigates modality imbalance and gradient conflicts in joint multimodal training by dynamically optimizing gradient weights and directions, ensuring both robustness on corner cases and competitive overall performance.



## 📖 Introduction / 简介

In multimodal learning, joint optimization often suffers from **modality imbalance**—a dominant modality (e.g., Vision) might yield exploding gradients that suppress or conflict with a weaker modality (e.g., Text/Audio). This leaves the model highly vulnerable to edge cases or long-tail samples where the dominant modality is noisy or absent.

**FairMGA** addresses this by introducing a "dynamic referee" during backpropagation:
- **Conflict Detection:** Real-time computation of directional cosine similarity (angles) between different modality gradients.
- **Fair Allocation:** Dynamically solves a constrained optimization problem to modulate gradient magnitudes and directions. 
- **Industrial Robustness:** Built-in numerical stability safeguards (`eps=1e-8`) and optimized data loading mechanisms, fully compatible with PyTorch Distributed Data Parallel (DDP).

在多模态联合训练中，强势模态的梯度往往会“淹没”或对冲弱势模态，导致模型在极端场景下极易崩溃。**FairMGA** 通过在反向传播中引入“梯度公平分配”约束，实时计算模态梯度夹角并求解带约束的优化问题，动态调整各模态权重。本框架不仅大幅下拉了困难样本的错误率，同时针对多卡分布式训练（DDP）进行了显存与死锁防御优化。

---

## 📂 Project Structure / 项目结构

```text
├── dataset/                  # Data loading and preprocessing pipelines
├── models/                   # Multimodal backbone networks and feature extractors
├── FairMGA.py                # Core implementation of FairMGA gradient modulation
├── min_norm_solvers.py       # Quad-programming / Minimum-norm solvers for gradient allocation
├── one_joint_loss.py         # Standard single joint loss baseline
├── uniform_baseline.py       # Uniformly-weighted multi-modal training baseline
├── weight_methods.py         # Common multi-task/multi-modal weight adjustment baselines
└── README.md                 # Project documentation
