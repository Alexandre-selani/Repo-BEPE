import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Modulos.Modelos import LeNet, LeNet_cac
from sklearn.manifold import TSNE
from Modulos.Datasets import Mnist_omni_loader
from Modulos.Utils import predict, NOMES

matplotlib.use("Agg")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

target_2_name = {i: f"Digit {i}" for i in range(10)}
target_2_name[-1] = "Omniglot"

cmap = plt.get_cmap("tab10")
target_2_color = {i: cmap(i) for i in range(10)}
target_2_color[-1] = "black"

data_manager = Mnist_omni_loader(bs=32, transform=NOMES.LENET_MNIST_OMNI_TRANSFORMS.value)
transform_val = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

save_dir = "/home/alexandreselani/Desktop/Experimento_mnist_omni/PCA_CAC"
os.makedirs(save_dir, exist_ok=True)
name = "MNIST_OMNI"
test_data = data_manager.load_test()
train_data = data_manager.load_train()

#----------------------------------------------------------------------------------------------------
#                                           PLOTTING CAC LOGITS
#----------------------------------------------------------------------------------------------------
model = LeNet_cac(num_classes=10)
model.skip_distance = True
model.load_state_dict(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet_cac/", "LeNet_mnist_omni_cac.pt")))
model.to(device=device)
model.eval()

y_true = []
for _, y in test_data:
    y_true.append(y)
y_true = torch.cat(y_true).cpu()

y_pred, __, test_logits = predict(test_data, model)
_, __, train_logits = predict(train_data, model)
test_logits, train_logits = test_logits.cpu(), train_logits.cpu()

# Scaling and PCA
scaler_cac = StandardScaler()
scaled_train_logits = scaler_cac.fit_transform(train_logits)
scaled_test_logits = scaler_cac.transform(test_logits)

pca_cac = PCA(n_components=2, random_state=42)
pca_cac.fit(scaled_train_logits)
test_logits_projected = pca_cac.transform(scaled_test_logits)

# Calculate Explained Variance
ev_cac = pca_cac.explained_variance_ratio_ * 100

plt.figure(figsize=(9, 7))
for klass in range(-1, 10, 1):
    class_logits = test_logits_projected[y_true == klass]
    if class_logits.shape[0] > 0:
        plt.scatter(
            class_logits[:, 0], class_logits[:, 1], 
            color=target_2_color[klass], label=target_2_name[klass], alpha=0.6
        )

# Fixed Title: Includes individual and total explained variance
plt.title(f"Logits Space - {name} (CAC)\nPC1: {ev_cac[0]:.2f}%, PC2: {ev_cac[1]:.2f}% (Total: {sum(ev_cac):.2f}%)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}_cac.png"))
plt.close()

#----------------------------------------------------------------------------------------------------
#                                           PLOTTING CE LOGITS
#----------------------------------------------------------------------------------------------------
model = LeNet(num_classes=10)
model.load_state_dict(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet/", "LeNet_mnist_omni.pt")))
model.to(device=device)
model.eval()

y_true = []
for _, y in test_data:
    y_true.append(y)
y_true = torch.cat(y_true).cpu()

# FIX: Added the missing train_logits prediction step for the CE model!
y_pred, __, test_logits = predict(test_data, model)
_, __, train_logits = predict(train_data, model)
test_logits, train_logits = test_logits.cpu(), train_logits.cpu()

# Scaling and PCA
scaler_ce = StandardScaler()
scaled_train_logits = scaler_ce.fit_transform(train_logits)
scaled_test_logits = scaler_ce.transform(test_logits)

pca_ce = PCA(n_components=2, random_state=42)
pca_ce.fit(scaled_train_logits)
test_logits_projected = pca_ce.transform(scaled_test_logits)

# Calculate Explained Variance
ev_ce = pca_ce.explained_variance_ratio_ * 100

plt.figure(figsize=(9, 7))
for klass in range(-1, 10, 1):
    class_logits = test_logits_projected[y_true == klass]
    if class_logits.shape[0] > 0:
        plt.scatter(
            class_logits[:, 0], class_logits[:, 1], 
            color=target_2_color[klass], label=target_2_name[klass], alpha=0.6
        )

# Fixed Title: Includes individual and total explained variance
plt.title(f"Logits Space - {name} (Cross Entropy)\nPC1: {ev_ce[0]:.2f}%, PC2: {ev_ce[1]:.2f}% (Total: {sum(ev_ce):.2f}%)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{name}.png"))
plt.close()