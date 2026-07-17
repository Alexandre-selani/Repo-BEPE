import torch
from Utils import AnaliseGraficaVal, train, eval, fix_random_seed
# Importe a classe que criamos anteriormente
from Datasets import CIFAR10Loader 

from ResNet18_backbone import ResNet18
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from Utils.Nomes import NOMES

# Configurações iniciais
fix_random_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

lr = 0.01
epochs = 35
bs = 256
num_classes = 10 # CIFAR10 tem 10 classes

# 1. Definição das Transformações
transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=4),  # standard for CIFAR
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])



# 2. Inicialização do Loader customizado
data_manager = CIFAR10Loader(root="./data/")

train_dataloader = data_manager.get_train_loader(transform=transform_train, bs=bs)
val_dataloader = data_manager.get_val_loader(transform=NOMES.CIFAR_RESNET18_VAL_TEST_TRANSFORMS.value, bs=bs)


# 4. Modelo, Critério e Otimizador
model = ResNet18(num_classes,weights=None)
model = model.to(device)
model_name = NOMES.RESNET18.value

optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                             momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = torch.nn.CrossEntropyLoss()

# 5. Log e Gráficos
grafico = AnaliseGraficaVal(f"{model_name}", "CIFAR10", dir="/home/alexandreselani/Desktop/Experimento_cifar10/")

# 6. Loop de Treinamento
for epoch in range(epochs):
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
model_dir = NOMES.RESNET18_CIFAR10.value

torch.save(model.state_dict(), model_dir)
print(f"Modelo salvo em {model_dir}")