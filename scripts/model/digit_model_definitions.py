"""Neural-network definitions for the digit classifiers."""

from torch import nn


class LeNet(nn.Module):
    """Classify 28 × 28 grayscale images using a LeNet-style CNN."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, images):
        return self.network(images)


class BespokeCNN_upsampling(nn.Module):
    """Enlarge digits before a longer geometric descent in spatial resolution.

    Upsampling gives learned convolutions more spatial room while channel counts
    grow. The hypothesis is that more learned processing along this descent may
    help unusual handwritten digits compared with early 28 -> 14 compression.
    Interpolation adds no new image information; any improvement is unproven.

    Input is normalized N x 1 x 28 x 28. Direct interpolation to 224 x 224
    represents three conceptual doublings, not three interpolation operations.
    Bilinear with align_corners=False is the accepted starting default, not an
    empirical winner; nearest and bicubic are also configurable. Bicubic can overshoot the
    input range. ReLU (default) or tanh follows every convolution.

    Each stage has TWO stride-one SAME convolutions, then 2 x 2 max pooling,
    stride 2, ceil_mode=True. Shapes below omit N and use C x H x W:
        2 x 224 x 224 -> pool 2 x 112 x 112
        4 x 112 x 112 -> pool 4 x 56 x 56
        8 x 56 x 56   -> pool 8 x 28 x 28
        16 x 28 x 28  -> pool 16 x 14 x 14
        32 x 14 x 14  -> pool 32 x 7 x 7
        64 x 7 x 7    -> pool 64 x 4 x 4
        128 x 4 x 4   -> pool 128 x 2 x 2
        256 x 2 x 2   -> pool 256 x 1 x 1
    Flatten -> Linear(256, 10) returns raw logits, without softmax.

    Even kernels require asymmetric SAME padding. Large kernels near the bottom
    use substantial zero padding. The resized input increases computation without
    increasing information. No batch normalization or global pooling is added.
    """

    def __init__(self, activation="relu", upsample_mode="bilinear"):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError("activation must be 'relu' or 'tanh'")
        if upsample_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("upsample_mode must be nearest, bilinear, or bicubic")
        self.activation = activation
        self.upsample_mode = upsample_mode
        resize_options = {} if upsample_mode == "nearest" else {"align_corners": False}
        self.resize = nn.Upsample(size=(224, 224), mode=upsample_mode, **resize_options)
        self.stages = nn.ModuleList()
        in_channels = 1
        for kernel, channels in zip(range(10, 2, -1), (2, 4, 8, 16, 32, 64, 128, 256)):
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel, stride=1, padding="same"),
                activations[activation](),
                nn.Conv2d(channels, channels, kernel, stride=1, padding="same"),
                activations[activation](),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ))
            in_channels = channels
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 10))

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("Expected images with shape (N, 1, 28, 28)")
        features = self.resize(images)
        for stage in self.stages:
            features = stage(features)
        return self.classifier(features)


class BespokeCNN(nn.Module):
    """Classify normalized 28 x 28 digits without resizing the input.

    This is a separate, untrained five-block architecture for a future comparison;
    the existing 224-upsample BespokeCNN_upsampling is unchanged. Each block has two stride-1
    SAME convolutions, an activation after each, then 2 x 2 max pooling at stride
    2 with ceil_mode=True. Activation is ReLU (default) or tanh.

    Shapes omit batch and use C x H x W. Both convolutions in each block keep
    the listed pre-pool size:
        16 x 28 x 28 -> pool  16 x 14 x 14
        32 x 14 x 14 -> pool  32 x  7 x  7
        64 x  7 x  7 -> pool  64 x  4 x  4
        128 x  4 x  4 -> pool 128 x  2 x  2
        256 x  2 x  2 -> pool 256 x  1 x  1
    Flatten 256 -> Linear(256, 10) returns raw logits, without model softmax.
    There is no resize, global average pooling, or batch normalization. Even
    kernels need asymmetric SAME zero padding; late kernels see substantial
    padding. No accuracy benefit has been established for this architecture.
    """

    def __init__(self, activation="relu"):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.activation = activation
        self.stages = nn.ModuleList()
        in_channels = 1
        for kernel, channels in zip((7, 6, 5, 4, 3), (16, 32, 64, 128, 256)):
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel, stride=1, padding="same"),
                activations[activation](),
                nn.Conv2d(channels, channels, kernel, stride=1, padding="same"),
                activations[activation](),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ))
            in_channels = channels
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 10))

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("Expected images with shape (N, 1, 28, 28)")
        features = images
        for stage in self.stages:
            features = stage(features)
        return self.classifier(features)
