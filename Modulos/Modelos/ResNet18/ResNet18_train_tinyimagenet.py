import torch
from Utils import AnaliseGraficaVal, train, eval, fix_random_seed
# Importe a classe que criamos anteriormente
from Datasets import Tinyimagenet_loader

from ResNet18_backbone import ResNet18
from torchvision.models import ResNet18_Weights
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from Utils.Nomes import NOMES

# Configurações iniciais
fix_random_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

lr = 0.001
epochs = 40
bs = 256
num_classes = 200

# 1. Definição das Transformações
transform_train  = transforms.Compose([
    transforms.RandomResizedCrop(64,scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),transforms.ColorJitter(
    brightness=0.2,  # randomly changes brightness by ±20%
    contrast=0.2,    # randomly changes contrast by ±20%
    saturation=0.2,  # randomly changes saturation by ±20%
    hue=0.05),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]),
])


# 2. Inicialização do Loader customizado
data_manager = Tinyimagenet_loader(root="~/.torchvision/tinyimagenet/",bs=bs)

train_dataloader = data_manager.load_train(transform=transform_train)
val_dataloader = data_manager.load_val(transform=NOMES.TINY_IMAGE_NET_RESNET18_VAL_TEST_TRANSFORMS.value)


# 4. Modelo, Critério e Otimizador
weights = ResNet18_Weights.IMAGENET1K_V1
model = ResNet18(num_classes,weights=weights)
model = model.to(device)
model_name = NOMES.RESNET18.value

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = torch.nn.CrossEntropyLoss()

# 5. Log e Gráficos
grafico = AnaliseGraficaVal(f"{model_name}", "TinyImageNet", dir="/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18")

# 6. Loop de Treinamento
for epoch in range(epochs):

    if epoch == 0:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True
    elif epoch == 10:
        for param in model.parameters():
            param.requires_grad = True


    model.train()
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
    
    model.eval()
    val_loss, val_acc = eval(val_dataloader, model, criterion)
    
    scheduler.step() 
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | lr = {optimizer.param_groups[0]['lr']:.6f}")
    print(f"VALIDATION    | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    grafico.addEpochVal(epoch, train_loss, train_acc, val_loss, val_acc)

grafico.mostraGraficoVal()

# 7. Salvamento
# Sugestão: mude o nome da constante no seu Enum para algo como RESNET18_CIFAR10
model_dir = NOMES.RESNET18_TINY_IMAGE_NET.value

torch.save(model.state_dict(), model_dir)
print(f"Modelo salvo em {model_dir}")