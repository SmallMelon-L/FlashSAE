# FlashSAE: Fast and Memory-Efficient Sparse Autoencoder with Sparse-Awareness

## Abstract

Training sparse autoencoders (SAEs) ares slow and memory-intensive in large latent size, as the time and memory complexity of conventional SAE training scale quadratically with the latent size. We argue that exploiting sparsity is the key to accelerating SAE training. In this work, we propose FlashSAE, a sparse-aware and exact training algorithm designed for  k-sparse autoencoders. FlashSAE leverages tensor sparsity during decoding and backpropagation to significantly reduce floating-point operations. In addition, it employs kernel fusion between the activation function and sparse tensor building, substantially lowering the overhead of building tensors with sparse layout. FlashSAE reduces both the computation and memory of decoding and backpropagation in k-sparse autoencoders from scaling linearly with the latent size to scaling linearly with the sparsity level k, enabling fast and memory-efficient training for SAEs with very wide latent spaces.

## Introduction

SAEs是一种在mechanistic interpretability领域中广泛使用的用于特征提取的工具。在实际应用中，根据Superposition Hypothesis，激活中蕴含的特征数量通常远远大于激活本身的维度。因此。SAEs的hidden layer通常具有数倍于输入维度的大小。这导致训练SAE所需的