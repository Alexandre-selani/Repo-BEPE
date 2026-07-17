
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Subset,ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Utils_OpenGan import *
import random
import os
from Feat_extraction import ResNet18_feature_extraction
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import KFold
device = 'cuda:0'
seed = 42
fix_random_seed(seed)

class RemmapedDataset(Dataset):
    def __init__(self, dataset, class_map):
        self.class_map = class_map
        self.dataset = dataset
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        X,y = self.dataset[idx]
        return X,self.class_map[int(y)]

class UUCDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        X,y = self.dataset[idx]
        return X,-1
def test(test_loader,detector):
    predicts=[]
    labels=[]

    for X, y in test_loader:
        
        #score eh a ativacao de todas as classes apos a openmax
        with torch.no_grad():
            score = detector(X.to(device))
            #print(score)
            max_values, predict = torch.max(score, dim=1)
        
        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
    
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    #ood_metrics.update(score[:,0],y)
    #metricas = metricasImplementadas(predict=predicts, label=labels)

    #print(ood_metrics.compute())
    print(predicts,labels)
    return predicts,labels
    
def train(train_loader,model,criterion,optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/(batch_idx+1), correct/total

def validation(val_loader,model,criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return val_loss/(batch_idx+1), correct/total

class ToUnknown(object):
    """
    Callable that returns a negative number, used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1
    
def main():
    
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

    mnist_train = MNIST(root="./data/", download=True, transform=transform, train=True)
    #mnist_test = MNIST(root="./data/", download=True, transform=transform, train=False)

    random_classes_train_splits = []
    
    for i in range(5):
        train_classes = random.sample(range(0,10), 6)
        random_classes_train_splits.append(set(train_classes))

    train_datasets_mnist = [[],[],[],[],[]]
    val_datasets_mnist_openset = [[],[],[],[],[]]

    for i,(y) in enumerate(mnist_train.targets):
        y = int(y)

        for j,train_split in enumerate(random_classes_train_splits):
            if y in train_split:
                train_datasets_mnist[j].append(i)
            else:
                val_datasets_mnist_openset[j].append(i)
    
    val_datasets_mnist_closedset =  [[],[],[],[],[]]
    for i, train_dataset in enumerate(train_datasets_mnist):
        train_dataset = np.array(train_dataset)

        kfold = KFold(n_splits=20, shuffle=True, random_state=seed)
        train_pos, val_pos = next(kfold.split(train_dataset))

        train_idx = train_dataset[train_pos]
        val_idx   = train_dataset[val_pos]

        train_datasets_mnist[i] = train_idx.tolist()
        val_datasets_mnist_closedset[i] = val_idx.tolist()

        val_datasets_mnist_openset[i] = np.random.choice(val_datasets_mnist_openset[i], size=len(val_idx), replace=False)
       
    print(len(train_datasets_mnist[0]))
    print(len(val_datasets_mnist_closedset[0]))
    print(len(val_datasets_mnist_openset[0]))

    lr=0.0001
    epochs = 13
    bs=64
    num_classes = 6
    import gc
    for iter in range(5):
        print(f"ITERACAO {iter}")
        print(f"Classes de treino: {random_classes_train_splits[iter]}")

        class_map_train = {cls:i for i,cls in enumerate(sorted(random_classes_train_splits[iter]))}
        
        
        print(class_map_train)
        torch.cuda.empty_cache()
        gc.collect()

        train_subset = Subset(mnist_train,train_datasets_mnist[iter])
        train_ds = RemmapedDataset(train_subset,class_map_train)
        train_labels_originais = [mnist_train.targets[i] for i in train_subset.indices]

        #print(train_labels_originais[:10])

        val_closedset_subset = Subset(mnist_train,val_datasets_mnist_closedset[iter])
        val_closed_ds = RemmapedDataset(val_closedset_subset,class_map_train)
        val_closed_labels_originais = [mnist_train.targets[i] for i in val_closedset_subset.indices]
        
        val_openset_subset = Subset(mnist_train,val_datasets_mnist_openset[iter])
        val_open_ds = UUCDataset(val_openset_subset)
        val_open_labels_originais = [mnist_train.targets[i] for i in val_openset_subset.indices]
        
        model = ResNet18_feature_extraction(num_classes=num_classes)
        model.model.to(device)

        train_dataloader = DataLoader(train_ds, batch_size=bs,shuffle=True)
        val_closedset_dataloader = DataLoader(val_closed_ds,batch_size=bs,shuffle=False)
        val_openset_dataloader = DataLoader(val_open_ds, batch_size=bs,shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    

        graficoValidacao = AnaliseGraficaVal(f"Feat extraction",f"Mnist_Iter_{iter}")
        for epoch in range(epochs):
        
            train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
            v_loss,v_acc = validation(val_closedset_dataloader,model.model,criterion)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
            #print(f"VALIDATION || Epoch {epoch+1}/{epochs} | Loss: {v_loss:.4f} | Acc: {v_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")

            graficoValidacao.addEpochVal(epoch=epoch,train_loss=train_loss,train_acc=train_acc,val_loss=v_loss,val_acc=v_acc)

        features_dir = f"./Features_extraidas/Mnist/Iter_{iter}/"
        
        graficoValidacao.mostraGraficoVal()

        if not os.path.exists(features_dir):
            os.makedirs(features_dir, exist_ok=True)

        torch.save(model.model.state_dict(), features_dir + "modelo.pth")
        model.save_features(train_dataloader,features_dir,"mnist_treino",train_labels_originais)
        model.save_features(val_openset_dataloader,features_dir,"mnist_val_openset",val_open_labels_originais)
        model.save_features(val_closedset_dataloader,features_dir,"mnist_val_closedset",val_closed_labels_originais)


        
        del model, criterion, optimizer, train_dataloader, val_openset_dataloader,val_closedset_dataloader
if __name__ == "__main__":
    main()