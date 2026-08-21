import sys
sys.path.insert(0, "/home/alexandreselani/Desktop/Modulos")

import torch
from Utils import AnaliseGraficaVal, train, eval,fix_random_seed, ToUnknown,NOMES
from Datasets import Mnist_omni_loader
from torch.utils.data import Subset,DataLoader,ConcatDataset
from Modelos import ResNet18
from torchvision.transforms import transforms
from torchvision.datasets import MNIST,Omniglot
import torch.nn as nn
import torch.optim as optim
import os

fix_random_seed(42)
device = "cuda:0"

lr=0.001
epochs = 20
bs=256

num_classes = 10

output_dir = "/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/"
os.makedirs(output_dir,exist_ok=True)

model = ResNet18(num_classes)
model = model.to(device)

#it is wrong, I should be using eval transforms for val set. It doesnt affect training.
data_manager_train = Mnist_omni_loader(bs,transform=NOMES.RESNET18_MNIST_OMNI_TRAIN_TRANSFORMS.value)
data_manager_val = Mnist_omni_loader(bs,transform=NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value)

train_dataloader= data_manager_train.load_train()
val_mnist_dataloader = data_manager_val.load_mnist_val()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# Cosine annealing: decai o lr suavemente de `lr` ate ~0 ao longo das epocas,
# refinando os pesos no fim do treino sem hiperparametro extra para ajustar.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

grafico = AnaliseGraficaVal("ResNet18","Mnist+Omni",dir=output_dir)

for epoch in range(epochs):

    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
    val_loss, val_acc = eval(val_mnist_dataloader,model,criterion)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
    print(f"VALIDATION || Epoch {epoch+1}/{epochs} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

    grafico.addEpochVal(epoch,train_loss,train_acc,val_loss,val_acc)

    # Depois dos prints: assim o lr exibido e o que foi usado nesta epoca.
    scheduler.step()

grafico.mostraGraficoVal()

model_dir = os.path.join("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/","ResNet18_mnist_omni.pt")

torch.save(model.state_dict(),model_dir)
print(f"Modelo salvo em {model_dir}")