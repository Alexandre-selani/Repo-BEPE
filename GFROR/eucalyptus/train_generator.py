import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import AlexNet_Weights
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from Datasets import Eucalyptus_openset_loader
from Utils import NOMES
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from model.vanilla_ae import VanillaAE320, VanillaAE,VanillaAE_eucalyptus
from model.utils import to_img

from Utils import fix_random_seed

fix_random_seed(42)


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_losses = []

    for batch, _ in tqdm(dataloader):
        batch = batch.float().to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def evaluate(model, dataloader, loss_fn, device):

    model.eval()
    eval_losses = []

    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.float().to(device)
            loss = loss_fn(model(batch), batch)
            eval_losses.append(loss.item())
    
    return sum(eval_losses) / len(eval_losses)
    

def vis(model, dataloader, filename, config, device):
    imgs = next(iter(dataloader))[0][:20]
    out = model(imgs.to(device))
    recons, org = to_img(out.cpu().data), to_img(imgs.cpu().data)
    merged = torch.stack((org,recons),dim=1).view(-1,3,32,32)
    save_image(merged, os.path.join(config.out_path, filename))

def main():

    config = {
        "batch_size": 32,
        "learning_rate":0.0001,
        "betas":(0.5, 0.999),
        "epochs": 50,
        "dataset":"dataset-1",
        "type":"train ecualyptus ae",
        "vis_only":True,
    }

    weights = AlexNet_Weights.IMAGENET1K_V1
    train_transform = weights.transforms()
    val_transform = weights.transforms()

#     train_transform = T.Compose([
#     T.Resize((32, 32)),
#     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     T.ToTensor()
# ])

#     val_transform = T.Compose([T.Resize(32), T.ToTensor()])
 
    data_manager = Eucalyptus_openset_loader(config["batch_size"])

    import gc
    for fold in range(5):
        best_val_loss = 999999

        gc.collect()
        torch.cuda.empty_cache()

        train_loader = data_manager.load_train(fold,train_transform)
        val_loader = data_manager.load_kkc_val(fold,val_transform)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VanillaAE_eucalyptus(512).to(device)
        loss_fn = torch.nn.L1Loss().to(device)
        #loss_fn = L1SSIMLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"], weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=4, threshold_mode='abs')

        ckpt_path = os.path.join("./ckpt/ae_eucalyptus", "eucalyptus")
        out_path = os.path.join("./output/ae_eucalyptus", "eucalyptus")


        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        train_losses = []
        val_losses = []

        for i in range(config["epochs"]):
            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            val_loss = evaluate(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)

            train_losses.extend(train_loss)
            val_losses.append(val_loss)

            print('epoch [{}/{}], lr ={:.4f} train loss:{:.4f}, val loss:{:.4f}'.format(i+1, config["epochs"], optimizer.param_groups[0]["lr"],sum(train_loss)/len(train_loss), val_loss))

            if (i+1) % 10 == 0:
                print(i+1)
                all_org = []
                all_recons = []
                with torch.no_grad():
                    for batch, _ in val_loader:
                        batch = batch.to(device)
                        out = model(batch)
                        all_org.append(to_img(batch.cpu().data, 224))
                        all_recons.append(to_img(out.cpu().data, 224))
                org = torch.cat(all_org, dim=0)
                recons = torch.cat(all_recons, dim=0)
                merged = torch.stack((org, recons), dim=1).view(-1, 3, 224, 224)
                image_path = os.path.join(out_path, f"fold_{fold}")
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                save_image(merged, os.path.join(image_path, "vanilla_recons_epoch{}.png".format(i+1)))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, os.path.join(ckpt_path, f"fold_{fold}.pth"))
                print("modelo salvo")

        optimizer.zero_grad()
        del train_loader,val_loader,model,optimizer,loss_fn
            


if __name__ == "__main__":
    main()
