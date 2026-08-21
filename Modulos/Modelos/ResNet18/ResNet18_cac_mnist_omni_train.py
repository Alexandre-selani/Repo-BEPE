import torch
from Utils import AnaliseGraficaVal, train_cac, eval_cac,fix_random_seed, ToUnknown,NOMES, CACLoss
from Datasets import Mnist_omni_loader
from torch.utils.data import Subset,DataLoader,ConcatDataset
from Modelos import ResNet18_cac
from torchvision.transforms import transforms
from torchvision.datasets import MNIST,Omniglot
import torch.nn as nn
import torch.optim as optim

fix_random_seed(42)
device = "cuda:0"

lr=0.001
epochs = 20
bs=256
lbda = 0.1
alpha = 10

num_classes = 10

model = ResNet18_cac(num_classes)
model = model.to(device)
anchors = torch.diag(torch.Tensor([alpha for i in range(num_classes)]))	
model.set_anchors(anchors)

data_manager = Mnist_omni_loader(bs,transform=NOMES.RESNET18_MNIST_OMNI_TRAIN_TRANSFORMS.value)

train_dataloader,val_mnist_dataloader = data_manager.load_train(), data_manager.load_mnist_val()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# Cosine annealing: decai o lr suavemente de `lr` ate ~0 ao longo das epocas.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

grafico = AnaliseGraficaVal(NOMES.RESNET18.value,"Mnist+Omni",dir="/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18_cac/")

for epoch in range(epochs):

    train_loss, train_acc = train_cac(train_dataloader, model, CACLoss, optimizer,num_classes,lbda)
    val_loss, val_acc = eval_cac(val_mnist_dataloader,model,CACLoss,num_classes,lbda)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
    print(f"VALIDATION || Epoch {epoch+1}/{epochs} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

    grafico.addEpochVal(epoch,train_loss,train_acc,val_loss,val_acc)

    # Depois dos prints: assim o lr exibido e o que foi usado nesta epoca.
    scheduler.step()

grafico.mostraGraficoVal()

model_dir = "/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18_cac/ResNet18_mnist_omni_cac.pt"

torch.save(model.state_dict(),model_dir)
print(f"Modelo salvo em {model_dir}")