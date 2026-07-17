from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from Utils import NOMES
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class PCA_plot:
    def __init__(self,dataset_name:str ,model_name:str,n_components:int = 2 ):
        self.n_components = n_components
        self.dataset = dataset_name
        self.model_name = model_name
        self.feats_dir = os.path.join(NOMES.FEATS_DIR.value,self.dataset,self.model_name)
        
    def _load_data(self,file_name: str) -> tuple[torch.Tensor,torch.Tensor]:
        """loads extracted features from a .pt file"""
        data = torch.load(os.path.join(self.feats_dir,file_name))
        feats,labels = data["features"],data["labels"]
        return feats,labels
    
    def plot_pca(self,fit_filename:str,transform_filenames:list[str],save_dir=None):
        #load data to fit scaler and pca
        fit_feats,fit_labels = self._load_data(fit_filename)

        #fit scaler and transform fit data
        scaler = StandardScaler().fit(fit_feats)
        fit_feats = scaler.transform(fit_feats)

        #fit pca
        pca = PCA(n_components=self.n_components,random_state=42).fit(fit_feats)

        #load all transform data
        all_labels = []
        all_feats = []
        for file in transform_filenames:
            feats, labels = self._load_data(file)
            all_labels.append(labels)
            all_feats.append(feats)

        all_labels = torch.cat(all_labels)
        all_feats = torch.cat(all_feats)

        #scale and pca transform data
        all_feats = scaler.transform(all_feats)
        all_feats = pca.transform(all_feats)

        ###### plot ###########
        unique_labels = np.unique(all_labels)
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))

        fig, ax = plt.subplots(figsize=(9, 7))
        for i, lbl in enumerate(unique_labels):
            mask = all_labels == lbl
            if lbl != -1:
                ax.scatter(
                    all_feats[mask, 0], all_feats[mask, 1],
                    label=str(lbl), s=8, alpha=0.6, color=cmap(i)
                )
            else:
                ax.scatter(
                    all_feats[mask, 0], all_feats[mask, 1],
                    label=str(lbl), s=8, alpha=0.6, color="black"
                )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
        ax.set_title(f"PCA – {self.model_name} Features ({self.dataset})")
        ax.legend(title="Class", markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()

        if(save_dir is None):
            fname = Path(f"~/Desktop/pca_plot_{self.model_name}_{self.dataset}.png").expanduser()
        else:
            fname = Path(save_dir).expanduser()
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"   Saved → {fname}")