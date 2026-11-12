# Model Implementation Options

<br>

## Overview
This table outlines possible approaches to the handwritten-digit classification task at different levels of implementation and training. It serves as a planning reference, but not all notebooks are required. Therefore, some notebooks may be postponed.

<br>

| Notebook | Option | Architecture | Reuses existing model implementation | Starts with pretrained weights | Trains on our data | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [02 — LeNet-style CNN](../notebooks/02_lenet_style_cnn.ipynb) | LeNet-style CNN | LeNet-style CNN | No | No | Yes | Use a simplified LeNet-style CNN. |
| [03 — Bespoke Upsampling CNN](../notebooks/03_bespoke_upsampling_cnn.ipynb) | Upsampling experiment | Eight two-convolution stages; 28 → 224 resize | No | No | Yes | Tests the upsampling approach with ReLU/tanh; kernels 10 through 3, channels 2 through 256, ceil-mode pooling. No batch normalization. |
| [04 — Bespoke Conventional CNN](../notebooks/04_bespoke_conventional_cnn.ipynb) | Conventional-style experiment | Five two-convolution blocks at native 28 × 28 input | No | No | Yes | Implemented and background-tested `BespokeCNN28`: kernels 7 through 3, channels 16 through 256; compare ReLU/tanh. |
| [05 — Bespoke Conventional CNN — Performance](../notebooks/05_bespoke_conventional_cnn_performance.ipynb) | Search for best performance | Same `BespokeCNN28` architecture as notebook 04 | Yes — notebook 04 model | No | Yes | Studies learning rates, fixed rotation augmentation, epoch budget and regularization. A performance experiment, not a third bespoke architecture. |
| TBD | Residual CNN / ResNet | CNN with shortcut connections | Optional | Optional | If pursued | Unpursued option: write a small residual network or adapt an existing ResNet for 28 × 28 grayscale images. |
| TBD | Fine-tuned pretrained model | Existing image-classification model | Yes | Yes | If pursued | Unpursued option: adapt a pretrained model’s input and classification layers, then fine-tune on our digits. |
