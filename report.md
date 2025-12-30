# FlashSAE: Fast and Memory-Efficient Exact SAE with Sparse-Awareness

## Abstract

Training Sparse Autoencoders (SAEs) at large model scales is computationally expensive, as the time complexity scales with both the input dimension and the number of latents. We argue that a key missing principle is to make SAE algorithms sparse-aware -- accounting for sparsity throughout the training process. We propose FlashSAE, a sparse-aware and exact SAE algorithm. FlashSAE exploits the sparsity of latent activations during decoding and backpropagation, and reduces memory usage and floating-point operations by fusing activation functions with kernels that construct sparse tensor layouts. As a result, for k-SAE—where each input activates k latents—the decoding and backward passes scale with k rather than the total number of latents. Empirically, FlashSAE significantly accelerates SAE training compared to existing approaches, without any loss in numerical accuracy.

## Introduction

Sparse Autoencoders (SAEs) and their variants, such as Top-K SAE, have become a core tool in mechanistic interpretability and feature discovery for large neural networks. By learning sparse latent representations aligned with meaningful internal features, SAEs enable fine-grained analysis of model behaviors and have been widely adopted in recent interpretability pipelines. As models and representation sizes continue to scale, modern SAEs are often trained with dictionary sizes ranging from tens of thousands to hundreds of thousands of latents.

Despite their conceptual simplicity, training SAEs at scale is computationally expensive. In practical settings, each input activates only a small number of latents, yet current training pipelines fail to exploit this sparsity effectively. Most existing implementations rely on dense tensor abstractions for both forward and backward passes, even when intermediate activations are highly sparse. As a result, decoding operations spend substantial compute on zero activations, while backpropagation still executes dense GEMM kernels whose cost scales with the full latent dimension. These inefficiencies lead to high latency, excessive memory consumption, and poor hardware utilization, making SAE training a system bottleneck in large-scale mechanistic interpretability workflows.

This mismatch between algorithmic sparsity and system-level execution highlights a missed optimization opportunity. While sparsity is fundamental to the modeling objective of SAEs, it is largely ignored by the underlying ML systems that execute them. In contrast, many recent system optimizations in deep learning focus on dense workloads, leaving sparse training regimes under-explored. For large SAEs, especially those with dictionary sizes of 32k to 512k, the cost of treating sparse activations as dense tensors becomes increasingly prohibitive.

In this work, we study how to make SAE training sparse-aware at both the algorithmic and system levels. We propose a system-level optimization framework that leverages the sparsity of latent activations throughout decoding and backpropagation. Our approach replaces dense computation with sparse-dense operations where appropriate, propagates gradients only along active features, and reduces memory overhead through compressed sparse representations. Furthermore, we observe that activation functions and sparse layout construction are tightly coupled, and introduce kernel fusion techniques to reduce intermediate memory traffic and kernel launch overhead.

We evaluate our approach through a detailed system-level analysis, measuring latency, GPU utilization and peak memory usage across realistic SAE training workloads. While our focus is on system performance, we also verify that sparse execution preserves the numerical behavior of the original dense implementation. Our results demonstrate that sparse-aware execution can significantly improve the efficiency of large-scale SAE training, offering practical guidance for building performant interpretability systems in modern ML frameworks.

## Preliminary



## Methodology

### 在forward decode阶段