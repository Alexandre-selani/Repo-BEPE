import torch
from Utils import AnaliseGraficaVal, train, eval, fix_random_seed
# Importe a classe que criamos anteriormente
from Datasets import Panicum_halfsize_loader
from Modelos.ResNet18_backbone import ResNet18
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from Utils.Nomes import NOMES
import os,gc
# Configurações iniciais
fix_random_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

N_FOLDS=5
save_dir = os.path.join("/home/alexandreselani/Desktop/Experimento_panicum",NOMES.RESNET18.value)
os.makedirs(save_dir,exist_ok=True)
gc.collect()

lr = 0.0001
epochs = 25
bs = 34
num_classes = 2

# 1. Definição das Transformações
transform_train = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90), # Essencial para drone
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val =transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], #igual imagenet
    std = [0.229, 0.224, 0.225])
])



# 2. Inicialização do Loader customizado
data_manager = Panicum_halfsize_loader(bs=bs)
weights = torch.load("/home/alexandreselani/Desktop/Modulos/Datasets/pesos/resnet18_weights_best_acc.tar")["model"]



# 4. Modelo, Critério e Otimizador

model_name = NOMES.RESNET18.value

for fold in range(N_FOLDS):
    gc.collect()
    torch.cuda.empty_cache()
      
    fold_dir = os.path.join(save_dir,f"Fold_{fold}")
    os.makedirs(fold_dir,exist_ok=True)

    grafico = AnaliseGraficaVal(f"{model_name}_Fold_{fold}","Panicum_plantnet_halfsize",dir=fold_dir)

    model = ResNet18(num_classes=2,weights=weights)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)

    train_dataloader,val_kkc_dataloader = data_manager.load_train(fold,transform_train), data_manager.load_kkc_val(fold,transform_val)


    for epoch in range(epochs):
        
        if epoch == 0:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.fc.parameters():
                param.requires_grad = True
        elif epoch == 10:
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True

        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
        val_loss, val_acc = eval(val_kkc_dataloader,model,criterion)
        #print(val_loss)
        scheduler.step(val_loss)
    
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
        print(f"VALIDATION | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

        grafico.addEpochVal(epoch,train_loss,train_acc,val_loss,val_acc)

    grafico.mostraGraficoVal()

    model_dir = os.path.join(fold_dir,f"{NOMES.RESNET18.value}_Panicum_fold_{fold}_plantnet.pt")

    torch.save(model.state_dict(),model_dir)
    print(f"Modelo salvo em {model_dir}")

    del model,train_dataloader,val_kkc_dataloader,optimizer,grafico,criterion,scheduler