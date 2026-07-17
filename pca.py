from Utils import PCA_plot,NOMES
import os
N_COMPONENTS    = 2
MODEL           = os.path.join(NOMES.RESNET18.value,"Fold_0/")
DATASET         = "Panicum_plantnet"


pca_plot = PCA_plot(DATASET,MODEL,N_COMPONENTS)

fit_data = "panicum_treino_features.pt"
transform_data = ["kkc_test_features.pt","uuc_test_features.pt"]

pca_plot.plot_pca(fit_data,transform_data,save_dir=".")