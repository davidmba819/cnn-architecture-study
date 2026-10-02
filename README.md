# CNN Architecture Study

An experimental study of CNN architectural design choices and their effect on image classification performance.

## Introduction

Convolutional Neural Networks (CNNs) can be designed in many different ways. Changes in depth, width, connectivity, feature extraction, normalization, and information flow can substantially change how a network learns.

The purpose of this project is to study these architectural behaviors experimentally rather than treating CNN architectures as fixed models that are simply trained and compared.

Five major CNN architectures were implemented and investigated:

1. VGG
2. Network in Network (NiN)
3. GoogLeNet / Inception
4. ResNet
5. DenseNet

Each architecture was studied according to the architectural ideas that define it. Instead of asking only which architecture produces the highest accuracy, the experiments investigate how specific architectural changes affect learning, model capacity, generalization, and classification performance.

## Project Objective

The primary objective of this project is to develop a practical understanding of CNN architecture through controlled experimentation.

The study investigates questions such as:

- What happens when a CNN becomes deeper?
- What happens when the number of feature channels is increased?
- How does model capacity affect training and generalization?
- What role does Batch Normalization play under controlled training conditions?
- How do 1×1 convolutions change the structure of a CNN?
- What is the effect of multi-branch feature extraction?
- How does residual connectivity affect the design of deeper networks?
- How does dense connectivity change the flow of information through a network?
- How does the DenseNet growth rate affect model capacity and performance?

The goal is therefore not simply to reproduce well-known CNN architectures. The goal is to understand the architectural principles behind them and observe their effects through controlled experiments.

## Scope of the Study

The project focuses specifically on architectural behavior.

For each architecture, selected structural parameters were varied while the main training conditions were kept controlled. This provides a basis for relating changes in model behavior to the architectural variable being investigated.

The major architectural factors studied include:

- Depth
- Width / number of channels
- Batch Normalization
- 1×1 convolutions
- Multi-branch feature extraction
- Residual connections
- Dense connectivity
- Growth rate

The experiments were not designed as a competition between architectures. Each architecture was treated as a separate case study with experiments designed around its own architectural characteristics.

The detailed implementation, training curves, experiment configurations, and individual model results are documented in the corresponding project notebooks.

# Dataset

The project uses the CIFAR-10 image classification dataset.

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- 10 classes
- RGB images
- Image resolution: 32 × 32 pixels

The original training set was divided into separate training and validation subsets. The official CIFAR-10 test set was kept untouched during training and validation and was used only for final evaluation.

The dataset is stored separately from the GitHub repository and is excluded from version control.

## Data Preparation

A common data preparation pipeline was used across the architecture experiments to keep the experimental conditions consistent.

Training data used:

- Random horizontal flipping
- Normalization

Validation and test data used:

- Normalization only

The normalization parameters were:

```text
Mean = [0.4914, 0.4822, 0.4465]

Standard deviation = [0.2470, 0.2435, 0.2616]

```

# Experimental Methodology

The central idea of the project was controlled architectural experimentation.

Rather than changing several aspects of a model at the same time, each experiment focused on a specific architectural property while keeping the remaining major conditions as consistent as possible.

For example, when studying depth, the number of blocks was changed while the main training configuration and other architectural settings were kept fixed.

When studying width, the number of channels was changed while the depth and other major settings were kept fixed.

This approach makes it easier to interpret the results because an observed performance change can be associated more directly with the architectural variable being investigated.

## Architecture-Specific Experiments

The experiments were designed around the defining characteristics of each architecture.

### VGG

VGG was used to study the effect of:

- Network depth
- Network width
- Batch Normalization

The VGG experiments focused on the idea of building deeper networks from repeated convolutional blocks using small 3×3 convolutions.

### Network in Network (NiN)

NiN was used to investigate:

- Network depth
- Network width
- The use of 1×1 convolutions
- Batch Normalization

The main architectural difference from VGG was the introduction of 1×1 convolutions within the convolutional blocks.

### GoogLeNet / Inception

GoogLeNet was used to study:

- Network depth
- Network width
- Multi-branch feature extraction
- The effect of the number of branches

The Inception structure allows different convolutional operations to process the same input in parallel before their outputs are concatenated.

### ResNet

ResNet was used to investigate:

- Network depth
- Network width
- Residual connectivity

The architecture uses residual blocks in which the input is combined with the output of the convolutional transformation through a shortcut connection.

The initial projection used a 1×1 convolution to transform the input channels before entering the residual blocks.

### DenseNet

DenseNet was used to investigate:

- Network depth
- Growth rate
- Dense connectivity

Instead of passing only the output of the previous layer to the next layer, each DenseNet layer receives the feature maps produced by earlier layers within the same dense block.

The growth rate controls the number of new feature maps produced by each dense layer.

## What Was Kept Consistent

Across the project, the following conditions were controlled as much as possible:

- Dataset
- Image resolution
- Data normalization
- Training/validation split
- Batch size
- Number of training epochs
- Optimizer
- Learning rate
- Momentum
- Loss function
- General evaluation procedure

The architectural components were then modified according to the experiment being performed.

This allowed the project to focus on the relationship between architectural design and observed model behavior.
