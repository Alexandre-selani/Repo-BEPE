import torch
from Utils import AnaliseGraficaVal, train, eval, fix_random_seed
# Importe a classe que criamos anteriormente
from Datasets import TinyImageNet_loader
from Modelos.ResNet18_backbone import ResNet18_tinyimgnet
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from Utils.Nomes import NOMES
import os, gc, copy

# Configurações iniciais
fix_random_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

N_SPLITS = 5
save_dir = os.path.join("/home/alexandreselani/Desktop/Experimento_tinyimgnet", NOMES.RESNET18.value)
os.makedirs(save_dir, exist_ok=True)
gc.collect()

lr = 8e-4
epochs = 100
warmup_epochs = 5
bs = 256
num_classes = 20  # classes conhecidas por split (protocolo OSR padrão do TinyImageNet)





def build_scheduler(optimizer, epochs, warmup_epochs):
    # AdamW do zero (stem recem-inicializado, sem pretrain) tende a dar passos ruidosos/grandes
    # nas primeiras epocas; um warmup linear curto antes do cosine decay evita essa instabilidade inicial.
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

# 1. Inicialização do Loader customizado
# As transformações (train/val) já são construídas internamente pelo loader,
# normalizadas com a média/desvio das classes conhecidas de cada split.
data_manager = TinyImageNet_loader(
    data_dir="/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200",
    splits_dir="/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits",
    batch_size=bs,
    image_size=64,
)

# 2. Modelo, Critério e Otimizador

model_name = NOMES.RESNET18.value

for split in range(N_SPLITS):
    gc.collect()
    torch.cuda.empty_cache()

    split_dir = os.path.join(save_dir, f"Split_{split}")
    os.makedirs(split_dir, exist_ok=True)

    grafico = AnaliseGraficaVal(f"{model_name}_Split_{split}", "TinyImageNet", dir=split_dir)

    model = ResNet18_tinyimgnet(num_classes=num_classes, weights=None)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs)

    # Augmentacao mais forte que a train_transform padrao do modulo (so nesse script):
    # com 10k imagens de treino/20 classes, o crop leve (0.8-1.0) + flip nao bastava.
    # Reusa a media/desvio do split para manter a normalizacao consistente com a validacao.
    mean, std = data_manager.norm_stats[split]
    strong_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])

    train_dataloader = data_manager.get_train_loader(split, strong_train_transform)
    val_kkc_dataloader = data_manager.get_val_known_loader(split, data_manager.eval_transforms[split])

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):

        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
        val_loss, val_acc = eval(val_kkc_dataloader, model, criterion)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
        print(f"VALIDATION | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

        grafico.addEpochVal(epoch, train_loss, train_acc, val_loss, val_acc)

    grafico.mostraGraficoVal()

    model_dir = os.path.join(split_dir, f"{NOMES.RESNET18.value}_TinyImageNet_split_{split}.pt")

    torch.save(best_state, model_dir)
    print(f"Modelo salvo em {model_dir} | Melhor acuracia de validacao: {best_val_acc:.4f}")

    del model, train_dataloader, val_kkc_dataloader, optimizer, grafico, criterion, scheduler
