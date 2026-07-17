from enum import Enum
import torchvision.transforms as transforms
from pathlib import Path


class NOMES(Enum):

    # ── Model Names ───────────────────────────────────────────────────────────
    RESNET18 = "ResNet18"
    ALEXNET  = "AlexNet"
    LENET    = "LeNet"

    # ── Feature Extraction ────────────────────────────────────────────────────
    FEATS_DIR = Path("/home/alexandreselani/Desktop/Features_extraidas/")

    # ── MNIST + Omniglot ──────────────────────────────────────────────────────
    MNIST_OMNI        = Path("Mnist_Omni/")
    MNIST_OMNI_SPLITS = Path("/home/alexandreselani/Desktop/Experimento_mnist_omni/Indices_mnist_omniglot/mnist_omni_splits.pt")

    RESNET18_MNIST_OMNI = Path("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18_mnist_omni.pt")
    ALEXNET_MNIST_OMNI  = Path("/home/alexandreselani/Desktop/Experimento_mnist_omni/AlexNet_mnist_omni.pt")
    LENET_MNIST_OMNI    = Path("/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet/LeNet_mnist_omni.pt")

    LENET_MNIST_OMNI_TRANSFORMS = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    RESNET18_MNIST_OMNI_TRANSFORMS = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    # ── Panicum ───────────────────────────────────────────────────────────────
    PANICUM_SPLITS = "/home/alexandreselani/Desktop/Experimento_panicum/Indices_panicum/panicum_splits.pt"
    PANICUM_SPLITS_HALFSIZE = "/home/alexandreselani/Desktop/Experimento_panicum/Indices_panicum_halfsize/panicum_splits.pt"
    PANICUM_RESNET = "/home/alexandreselani/Desktop/Experimento_panicum/ResNet18"
    
    PANICUM_PLANTNET_VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], #igual imagenet
        std = [0.229, 0.224, 0.225])
    ])


    # ── Tiny ImageNet ─────────────────────────────────────────────────────────
    TINY_IMAGE_NET        = Path("Tinyimgnet/")
    RESNET18_TINY_IMAGE_NET = Path("/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/ResNet18_tinyimgnet.pt")

    TINY_IMAGE_NET_RESNET18_VAL_TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]),
    ])

    # ── CIFAR ─────────────────────────────────────────────────────────────────
    CIFAR_RESNET18_VAL_TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    RESNET18_CIFAR10 = Path("/home/alexandreselani/Desktop/Experimento_cifar10/resnet18_cifar10.pt")