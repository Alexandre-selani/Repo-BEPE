import torch
from Modulos.Modelos import ResNet18_cac,ResNet18
from Modulos.Datasets import Panicum_halfsize_loader
from Modulos.Utils import predict
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
device = "cuda:0"
import os

target_2_color = {0:"blue",
                 1:"orange",
                 -1:"red"}
target_2_name = {0:"Ground",
                 1:"Healthy",
                 -1:"Panicum"}

data_manager = Panicum_halfsize_loader(bs=32)
transform_val =transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], #igual imagenet
    std = [0.229, 0.224, 0.225])
])
for fold in range(5):
    test_data = data_manager.load_test(fold,transform_val)

    model = ResNet18_cac(num_classes=2)
    model.skip_distance=True
    model.load_state_dict(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum_cac/ResNet18/", f"Fold_{fold}/ResNet18_Panicum_cac_fold_{fold}_plantnet.pt")))
    model.to(device=device)
    model.eval()

    # model = ResNet18(num_classes=2)
    # model.load_state_dict(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum/ResNet18/", f"Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt")))
    # model.to(device=device)
    # model.eval()

    y_true = []
    for _,y in test_data:
        y_true.append(y)
    
    y_true = torch.cat(y_true)
    
    y_pred, __, test_logits = predict(test_data, model)
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    test_logits = test_logits.cpu() # Move logits to CPU once here

    # Create a clean plot for this fold
    plt.figure(figsize=(8, 6))

    for klass in [-1, 0, 1]:  # Cleaner syntax for your three classes
        # Filter logits belonging to the current class
        class_logits = test_logits[y_true == klass]
        
        # Check if there are actually samples for this class in this fold
        if class_logits.shape[0] > 0:
            print(f"Fold {fold} | Class {klass} ({target_2_name[klass]}): Plotting {class_logits.shape[0]} points.")
            
            # FIX: class_logits[:, 0] is X-axis (Feature 1), class_logits[:, 1] is Y-axis (Feature 2)
            plt.scatter(
                class_logits[:, 0], 
                class_logits[:, 1], 
                color=target_2_color[klass],
                label=target_2_name[klass],
                alpha=0.6 # Adds transparency to see overlapping points clearly
            )
        else:
            print(f"Fold {fold} | Class {klass} ({target_2_name[klass]}): No samples found.")
    
    # Beautify and save the plot
    plt.title(f"Logits Space - Fold {fold}")
    plt.xlabel("Logit Dimension 1")
    plt.ylabel("Logit Dimension 2")
    plt.legend() # Uses your target_2_name labels automatically
    plt.savefig(f"panicum_cac_fold{fold}.png")
    plt.close()

for fold in range(5):
    test_data = data_manager.load_test(fold,transform_val)

    model = ResNet18(num_classes=2)
  
    model.load_state_dict(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum/ResNet18/", f"Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt")))
    model.to(device=device)
    model.eval()

    # model = ResNet18(num_classes=2)
    # model.load_state_dict(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum/ResNet18/", f"Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt")))
    # model.to(device=device)
    # model.eval()

    y_true = []
    for _,y in test_data:
        y_true.append(y)
    
    y_true = torch.cat(y_true)
    
    y_pred, __, test_logits = predict(test_data, model)
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    test_logits = test_logits.cpu() # Move logits to CPU once here

    # Create a clean plot for this fold
    plt.figure(figsize=(8, 6))

    for klass in [-1, 0, 1]:  # Cleaner syntax for your three classes
        # Filter logits belonging to the current class
        class_logits = test_logits[y_true == klass]
        
        # Check if there are actually samples for this class in this fold
        if class_logits.shape[0] > 0:
            print(f"Fold {fold} | Class {klass} ({target_2_name[klass]}): Plotting {class_logits.shape[0]} points.")
            
            # FIX: class_logits[:, 0] is X-axis (Feature 1), class_logits[:, 1] is Y-axis (Feature 2)
            plt.scatter(
                class_logits[:, 0], 
                class_logits[:, 1], 
                color=target_2_color[klass],
                label=target_2_name[klass],
                alpha=0.6 # Adds transparency to see overlapping points clearly
            )
        else:
            print(f"Fold {fold} | Class {klass} ({target_2_name[klass]}): No samples found.")
    
    # Beautify and save the plot
    plt.title(f"Logits Space - Fold {fold}")
    plt.xlabel("Logit Dimension 1")
    plt.ylabel("Logit Dimension 2")
    plt.legend() # Uses your target_2_name labels automatically
    plt.savefig(f"panicum_NOcac_fold{fold}.png")
    plt.close()






    



    

    