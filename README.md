# DogeNet

Have you ever wondered what a neural network architecture would look like if it were designed by dogs for dogs? If you answered "woof" to that question then you are in for a tasty treat my fellow furry friend! In this example project, we show how we can adapt the VGG and ResNet architectures used in computer vision for localizing and classifying a variety of dog breeds. We use the Stanford Dogs dataset consisting of images and annotations of 120 classes of dog breed to train and test our model. In addition, we make some simple tweaks to the aforementioned network architectures that improve overall training performance while maintaining the "spirit" of the original design.

## Download the Repository
Clone the project locally using git:

```
git clone https://github.com/andrewdalpino/DogeNet
```

## Requirements

- [Python](https://www.python.org/) 3.10 or later
- A CPU and at least 8G of available memory

## Recommended

- A CUDA-enabled GPU with 12G of VRAM or more

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. I recommend using a virtual environment such as venv to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Tutorial

Stay tuned ...

## References:
>- Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR, 2015.
>- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. Microsoft Research, 2015.
>- Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.
>- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.

## License
The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).