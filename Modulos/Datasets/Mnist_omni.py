from Utils import *
from torchvision.datasets import MNIST, Omniglot
from torch.utils.data import ConcatDataset

fix_random_seed(42)

mnist_train = MNIST(root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data/",download=True,transform=NOMES.RESNET18_MNIST_OMNI_TRANSFORMS,train=True)
mnist_test = MNIST(root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data/",download=True,transform=NOMES.RESNET18_MNIST_OMNI_TRANSFORMS,train=False)
test_omniglot = Omniglot(root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data/",download=True,transform=NOMES.RESNET18_MNIST_OMNI_TRANSFORMS,target_transform=ToUnknown())
test_omniglot_reduzido = random_dataset(test_omniglot,10000)
val_omniglot = Omniglot(root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data/",download=True,transform=NOMES.RESNET18_MNIST_OMNI_TRANSFORMS,target_transform=ToUnknown(),background=False)

train_dataset,mnist_val_dataset = validation_split(0.1,mnist_train)
val_omniglot_reduzido = random_dataset(val_omniglot,len(mnist_val_dataset))

grid_search_val_dataset = ConcatDataset([mnist_val_dataset,val_omniglot_reduzido])
test_dataset = ConcatDataset([test_omniglot_reduzido,mnist_test])

splits = {
"mnist_train_idx": train_dataset.indices,
"mnist_val_idx": mnist_val_dataset.indices,
"omniglot_test_idx": test_omniglot_reduzido.indices,
"omniglot_val_idx": val_omniglot_reduzido.indices,
}

for x in splits.items():
    print(len(x[1]))

torch.save(splits, NOMES.MNIST_OMNI_SPLITS)