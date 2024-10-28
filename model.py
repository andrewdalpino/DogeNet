from torch.nn import (
    Module, Sequential, Conv2d, MaxPool2d, AvgPool2d, Linear, GELU,
    Identity, Flatten, BatchNorm1d, BatchNorm2d, LogSoftmax,
)

class ResDoge50(Module):
    """
    A variation on the ResNet50 architecture with added regression head for localization.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.body = Sequential(
            Conv7x7Block(3, 64, stride=2),
            MaxPool2d(3, 2),
            BottleneckBlock(64, 256),
            BottleneckBlock(256, 256),
            BottleneckBlock(256, 256),
            BottleneckBlock(256, 512, stride=2),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 1024, stride=2),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024, stride=2),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
        )

        self.classifier_head = Sequential(
            BottleneckBlock(1024, 2048, stride=2),
            BottleneckBlock(2048, 2048),
            BottleneckBlock(2048, 2048),
            AvgPool2d(2, 2),
            Flatten(),
            SoftmaxClassifier(8192, num_classes),
        )

        self.box_head = Sequential(
            BottleneckBlock(1024, 1024, stride=2),
            BottleneckBlock(1024, 1024),
            AvgPool2d(2, 2),
            Flatten(),
            LinearRegressor(4096, 4),
        )

    @property
    def num_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x):
        x = self.body(x)

        y1 = self.classifier_head(x)
        y2 = self.box_head(x)

        return y1, y2

class ResDoge34(Module):
    """
    A quasi-ResNet34 architecture with additional regression head for bounding box estimation.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.body = Sequential(
            Conv7x7Block(3, 64, stride=2),
            MaxPool2d(3, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )

        self.classifier_head = Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AvgPool2d(2, 2),
            Flatten(),
            SoftmaxClassifier(2048, num_classes),
        )

        self.box_head = Sequential(
            ResidualBlock(256, 256, stride=2),
            ResidualBlock(256, 256),
            AvgPool2d(2, 2),
            Flatten(),
            LinearRegressor(1024, 4),
        )

    @property
    def num_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x):
        x = self.body(x)

        y1 = self.classifier_head(x)
        y2 = self.box_head(x)

        return y1, y2

class VGGDoge(Module):
    """
    A VGG-like architecture with added regression head for bounding box estimation.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.body = Sequential(
            Conv2XBlock(3, 64),
            MaxPool2d(2, 2),
            Conv2XBlock(64, 128),
            MaxPool2d(2, 2),
            Conv2XBlock(128, 256),
            Conv2XBlock(256, 256),
            MaxPool2d(2, 2),
            Conv2XBlock(256, 256),
            Conv2XBlock(256, 256),
            MaxPool2d(2, 2),
            Conv2XBlock(256, 512),
            Conv2XBlock(512, 512),
            MaxPool2d(2, 2),
            Conv2XBlock(512, 512),
            Conv2XBlock(512, 512),
            MaxPool2d(2, 2),
            Flatten(),
        )

        self.classifier_head = Sequential(
            FullyConnected(4608, 2048),
            FullyConnected(2048, 2048),
            SoftmaxClassifier(2048, num_classes),
        )

        self.box_head = Sequential(
            FullyConnected(4608, 2048),
            FullyConnected(2048, 1024),
            FullyConnected(1024, 512),
            LinearRegressor(512, 4),
        )

    @property
    def num_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x):
        x = self.body(x)

        y1 = self.classifier_head(x)
        y2 = self.box_head(x)

        return y1, y2

class Conv7x7Block(Module):
    """The first layer of residual style networks."""
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()

        self.layers = Sequential(
            Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=7,
                stride=stride,
                padding=1,
                bias=False
            ),
            BatchNorm2d(channels_out),
            GELU(),
        )

    def forward(self, x):
        return self.layers.forward(x)

class ResidualBlock(Module):
    """A basic residual block."""
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()

        self.residual = Sequential(
            Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(channels_out),
            GELU(),
            Conv2d(
                in_channels=channels_out,
                out_channels=channels_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(channels_out),
        )

        if stride == 1 and channels_in == channels_out:
            shortcut = Sequential(Identity())
        else:
            shortcut = Sequential(
                Conv2d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False
                ),
            )

        self.shortcut = shortcut
        self.batch_norm = BatchNorm2d(channels_out)
        self.activation = GELU()

    def forward(self, x):
        residual = self.residual.forward(x)

        x_hat = self.shortcut(x)

        z = self.batch_norm(x_hat + residual)

        out = self.activation(z)

        return out

class BottleneckBlock(Module):
    """
    A type of residual layer that aims to reduce the number of parameters
    while maintaining the representational capacity of a 2X Residual block
    using projections.
    """
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()

        self.residual = Sequential(
            Conv2d(
                in_channels=channels_in,
                out_channels=channels_in,
                kernel_size=1,
                stride=stride,
                padding='valid',
                bias=False,
            ),
            BatchNorm2d(channels_in),
            GELU(),
            Conv2d(
                in_channels=channels_in,
                out_channels=channels_in,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
            BatchNorm2d(channels_in),
            GELU(),
            Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=1,
                padding='valid',
                bias=False,
            ),
        )

        if stride == 1 and channels_in == channels_out:
            shortcut = Sequential(Identity())
        else:
            shortcut = Sequential(
                Conv2d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False
                ),
            )

        self.shortcut = shortcut
        self.batch_norm = BatchNorm2d(channels_out)
        self.activation = GELU()

    def forward(self, x):
        residual = self.residual.forward(x)

        x_hat = self.shortcut(x)

        z = self.batch_norm(x_hat + residual)
        
        out = self.activation(z)

        return out

class Conv2XBlock(Module):
    """Two convolutional layers applied sequentially with batch norm."""
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()

        self.layers = Sequential(
            Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=3,
                stride=stride,
                padding='same',
                bias=False,
            ),
            BatchNorm2d(channels_out),
            GELU(),
            Conv2d(
                in_channels=channels_out,
                out_channels=channels_out,
                kernel_size=3,
                stride=stride,
                padding='same',
                bias=False,
            ),
            BatchNorm2d(channels_out),
            GELU(),
        )

    def forward(self, x):
        return self.layers.forward(x)

class FullyConnected(Module):
    def __init__(self, input_features: int, num_neurons: int):
        super().__init__()

        self.layers = Sequential(
            Linear(
                in_features=input_features,
                out_features=num_neurons,
                bias=False,
            ),
            BatchNorm1d(num_features=num_neurons),
            GELU(),
        )

    def forward(self, x):
        return self.layers.forward(x)

class SoftmaxClassifier(Module):
    def __init__(self, input_features: int, num_classes: int):
        super().__init__()

        self.layers = Sequential(
            Linear(
                in_features=input_features,
                out_features=num_classes,
                bias=True,
            ),
            LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.layers.forward(x)

class LinearRegressor(Module):
    def __init__(self, input_features: int, num_outputs: int):
        super().__init__()

        self.layers = Sequential(
            Linear(
                in_features=input_features,
                out_features=num_outputs,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.layers.forward(x)