import torch
from Utils import AnaliseGraficaVal, train, eval, fix_random_seed
# Importe a classe que criamos anteriormente
from Datasets import Eucalyptus_closedset_loader
from Modelos.ResNet18_32x32_backbone import ResNet18_32x32
from Modelos.AlexNet_backbone import Alexnet
from torchvision import transforms
from torchvision.models import AlexNet_Weights
import torch.nn as nn
import torch.optim as optim
from Utils.Nomes import NOMES
import gc
import os

# Configurações iniciais
fix_random_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

lr = 0.0001
epochs = 30
bs = 128
num_classes = 3
n_folds = 5
dataset= "dataset-1"

weights = AlexNet_Weights.IMAGENET1K_V1


# 2. Inicialização do Loader customizado
data_manager = Eucalyptus_closedset_loader(bs=bs)
model_name = "AlexNet"

for fold in range(n_folds):
    torch.cuda.empty_cache()
    
    gc.collect()

    grafico = AnaliseGraficaVal(f"{model_name}_fold_{fold}", "Eucalyptus closed set", dir=f"/home/alexandreselani/Desktop/Eucalyptus/ClosedSet/Graficos/{dataset}/")

    # 4. Modelo, Critério e Otimizador
    model = Alexnet(num_classes,weights=weights)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader = data_manager.load_train(transform=weights.transforms(),fold=fold)
    
    val_dataloader = data_manager.load_val(transform=weights.transforms(),fold=fold)

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
        
        model.eval()
        val_loss, val_acc = eval(val_dataloader, model, criterion)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | lr = {optimizer.param_groups[0]['lr']:.6f}")
        print(f"VALIDATION    | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        grafico.addEpochVal(epoch, train_loss, train_acc, val_loss, val_acc)

    grafico.mostraGraficoVal()

    # 7. Salvamento
    # Sugestão: mude o nome da constante no seu Enum para algo como RESNET18_CIFAR10
    model_dir = f"/home/alexandreselani/Desktop/Eucalyptus/ClosedSet/Models/{dataset}/"
    os.makedirs(model_dir,exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}AlexNet_fold_{fold}.pt")
    print(f"Modelo salvo em {model_dir}")

    del model,optimizer,criterion,grafico,train_dataloader,val_dataloader