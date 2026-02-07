<div align='center'>
 
# Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization

<!-- [![preprint](https://img.shields.io/badge/arXiv-2510.00761-B31B1B)](https://arxiv.org/abs/2510.00761) -->

</div>

## Abstract
Zeroth-order (ZO) optimization provides a gradient-free alternative to first-order (FO) methods by estimating gradients via finite differences of function evaluations, and has recently emerged as a memory-efficient paradigm for fine-tuning large-scale models by avoiding backpropagation. However, ZO optimization has a fundamental tension between accuracy and query efficiency. In this work, we show that ZO optimization can be substantially improved by unifying two complementary principles: (i) a projection-based subspace view that reduces gradient estimation variance by exploiting the intrinsic low-rank structure of model updates, and (ii) Muon-style spectral optimization that applies gradient orthogonalization to extract informative spectral structure from noisy ZO gradients. These findings form a unified framework of subspace gradient orthogonalization, which we instantiate in a new method, **ZO-Muon**, admitting a natural interpretation as a low-rank Muon optimizer in the ZO setting. Extensive experiments on large language models (LLMs) and vision transformers (ViTs) demonstrate that ZO-Muon significantly accelerates convergence and achieves a win–win improvement in accuracy and query/runtime efficiency. Notably, compared to the popular MeZO baseline, ZO-Muon requires only 24.7% of the queries to reach the same SST-2 performance for LLM fine-tuning, and improves accuracy by 25.1% on ViT-B fine-tuning on CIFAR-100.
<!-- 
## Cite This Work
```
@article{lang2025downgrade,
  title={Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning},
  author={Lang, Yicheng and Zhang, Yihua and Fan, Chongyu and Wang, Changsheng and Jia, Jinghan and Liu, Sijia},
  journal={arXiv preprint arXiv:2510.00761},
  year={2025}
}
``` -->