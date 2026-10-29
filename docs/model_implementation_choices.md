# Model Implementation Options

<br>

## Overview
This table outlines possible approaches to the handwritten-digit classification task at different levels of implementation and training. It serves as a planning reference, but not all notebooks are required. Therefore, some notebooks may be postponed.

<br>

| Notebook | Option | Architecture | Reuses existing model implementation | Starts with pretrained weights | Trains on our data | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [02 — LeNet-style CNN](../notebooks/02_lenet_style_cnn.ipynb) | LeNet-style CNN | LeNet-style CNN | No | No | Yes | Use a simplified LeNet-style CNN. |
| TBD | Bespoke CNN | Bespoke CNN | No | No | Yes | Choose our own filter sizes, filter counts, pooling, and classification layers, and include batch normalization. |
| TBD | Residual CNN / ResNet | CNN with shortcut connections | Optional | Optional | Yes | Write a small residual network or adapt an existing ResNet implementation for 28 × 28 grayscale images. |
| TBD | Fine-tuned pretrained model | Existing image-classification model | Yes | Yes | Yes | Adapt the input and classification layers of a pretrained model, then fine-tune on our digits. |

## References

- [LeNet](https://en.d2l.ai/chapter_convolutional-neural-networks/lenet.html)
- [Batch normalization](https://en.d2l.ai/chapter_convolutional-modern/batch-norm.html)
- [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
- [Torchvision models and pretrained weights](https://docs.pytorch.org/vision/master/models.html)
