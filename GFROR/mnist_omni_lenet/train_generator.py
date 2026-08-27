import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from Datasets import Mnist_omni_loader
from Utils import NOMES
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from model.vanilla_ae import VanillaAE
from model.utils import to_img


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
        "batch_size": 256,
        "learning_rate":1e-4,
        "betas":(0.5, 0.999),
        "epochs": 20,
        "split": 3,
        "ckpt_period":25,
        "type":"train mnist ae",
        "vis_only":True,
    }

    train_transform = T.Compose([T.Resize(32),T.RandomHorizontalFlip(),T.Grayscale(num_output_channels=3), T.ToTensor(), #T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    val_transform = T.Compose([T.Resize(32),T.Grayscale(num_output_channels=3),T.ToTensor(), #T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    data_manager_train = Mnist_omni_loader(config["batch_size"],train_transform)
    data_manager_val = Mnist_omni_loader(config["batch_size"], val_transform)
   
    train_loader = data_manager_train.load_train()
    val_loader = data_manager_val.load_mnist_val()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaAE().to(device)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, threshold_mode='abs')

    ckpt_path = os.path.join("./ckpt/ae_mnist_omni", "Mnist_omni")
    out_path = os.path.join("./output/ae_mnist_omni", "Mnist_omni")


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

        print('epoch [{}/{}], train loss:{:.4f}, val loss:{:.4f}'.format(i+1, config["epochs"], sum(train_loss)/len(train_loss), val_loss))

        imgs = next(iter(val_loader))[0][:20]
        out = model(imgs.to(device))
        recons = to_img(out.cpu().data)
        org = to_img(imgs.cpu().data)
        merged = torch.stack((org,recons),dim=1).view(-1,3,32,32)
        save_image(merged, os.path.join(out_path, "vanilla_recons_epoch{}.png".format(i+1)))

        
        torch.save(model, os.path.join(ckpt_path, "ckpt.pth"))


if __name__ == "__main__":
    main()
