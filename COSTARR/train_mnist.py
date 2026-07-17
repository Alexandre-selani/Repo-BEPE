import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from funcs import *

# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

# ─── Data ─────────────────────────────────────────────────────────────────
# MNIST: 1 channel, 28x28 → precisa adaptar para ResNet (3 canais, 224x224)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # 1 canal → 3 canais
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307]*3, std=[0.3081]*3),
])

train_set = torchvision.datasets.MNIST(root="./data", train=True,
                                       download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root="./data", train=False,
                                       download=True, transform=transform)

omniglot_full = torchvision.datasets.Omniglot(root="./data",
                                       download=True, transform=transform,
                                       target_transform=ToUnknown())

# Apenas 10.000 imagens do Omniglot no test_loader
omniglot_10k, _ = torch.utils.data.random_split(omniglot_full, [10000, len(omniglot_full) - 10000])

# Concatenacao: test_set (MNIST) + 10k Omniglot (targets = -1)
combined_test = torch.utils.data.ConcatDataset([test_set, omniglot_10k])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(combined_test, batch_size=BATCH_SIZE, shuffle=False)

# ─── Model ────────────────────────────────────────────────────────────────
class ResNetFeaturizer(nn.Module):
    """Wrapper que retorna (logits, features) no forward."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove o FC para extrair features separadamente
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])  # até o avgpool
        self.fc = self.backbone.fc  # (512, num_classes)

    def forward(self, x):
        # Features: (batch, 512, 1, 1) → (batch, 512)
        feats = self.features(x).flatten(1)
        logits = self.fc(feats)
        return logits, feats

    def getPerClassWeights(self):    # Obtem o ultimo modulo (camada) do modelo
        last_layer = list(self.backbone.modules())[-1]
        with torch.no_grad():
            return last_layer.weight.detach()
model = ResNetFeaturizer(num_classes=10).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─── Treino ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Train")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, feats = model(x)  # ← desempacota (logits, features)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct   += (logits.argmax(1) == y).sum().item()
        total     += x.size(0)

        pbar.set_postfix(loss=loss.item(), acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total

# ─── Avaliação ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, feats = model(x)  # ← desempacota (logits, features)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct   += (logits.argmax(1) == y).sum().item()
        total     += x.size(0)
    return total_loss / total, correct / total

def train():
    print(f"Device: {DEVICE}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        )

        costarrFit(model,train_loader,"mnist_costarr")

    # ─── Salvar ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "resnet_mnist.pth")
    print("Modelo salvo em resnet_mnist.pth")

def test():
    train_calc = torch.load("mnist_costarr.pt",weights_only=False)
    model.load_state_dict(torch.load("resnet_mnist.pth"))
    y_pred,y_true = costarrPredict(model,test_loader,train_calc)
    print(accuracy_score(y_pred,y_true))
test()
