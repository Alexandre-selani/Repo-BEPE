from Utils.LinewithSTD import *
import numpy as np
from pathlib import Path

MIN_IDX = 30
MAX_IDX = 60

x_ticks = np.arange(0.3,0.6,0.01)
for i,x in enumerate(x_ticks):
    print(x)
    x_ticks[i] = round(x,2)

y_ticks = np.arange(0,1.1,0.1)
for i,y in enumerate(y_ticks):
    y_ticks[i] = round(y,2)

format = "pdf"
plt.rcParams.update({'font.size': 20})

cfg = PlotConfig("Epsilon","F1 score",x_ticks,y_ticks,["Test","Validation"],"F1 score mean and standard devation on Test/Val sets per Epsilon - OpenGAN (Panicum dataset)")
TEST_FOLDER = Path("/home/alexandreselani/Desktop/OpenGan/Resultados/Panicum_plantnet/ResNet18/Test/")
VAL_FOLDER = Path("/home/alexandreselani/Desktop/OpenGan/Resultados/Panicum_plantnet/ResNet18/Val/")
BASE_SAVE_DIR = Path("~/Desktop/Plot/OpenGan/").expanduser()
MEAN_COL_NAME = "f1_macro_medio"
STD_COL_NAME = "f1_macro_std"

plot_pipeline = PlottingPipeline(TEST_FOLDER,VAL_FOLDER,BASE_SAVE_DIR,MEAN_COL_NAME,STD_COL_NAME)
plot_pipeline.run(cfg,show_std=True,min_idx=MIN_IDX,max_idx=MAX_IDX,format=format)

cfg = PlotConfig("Epsilon","F1 score",x_ticks,y_ticks,["Test","Validation"],"F1 score per Epsilon on Fold {idx} - OpenGAN (Panicum dataset)")
TEST_FOLDER = Path("/home/alexandreselani/Desktop/OpenGan/Resultados/Panicum_plantnet/ResNet18/Test/Folds")
VAL_FOLDER = Path("/home/alexandreselani/Desktop/OpenGan/Resultados/Panicum_plantnet/ResNet18/Val/Folds")
BASE_SAVE_DIR = Path("~/Desktop/Plot/OpenGan/").expanduser()
COL_NAME = "f1_macro"

plot_pipeline = PlottingPipeline(TEST_FOLDER,VAL_FOLDER,BASE_SAVE_DIR,COL_NAME,None)
plot_pipeline.run(cfg,show_std=False,min_idx=MIN_IDX,max_idx=MAX_IDX,format=format)

##################### OPENMAX ############################

x_ticks = np.arange(0,51,5)


cfg = PlotConfig("Tail","F1 score",x_ticks,y_ticks,["Test","Validation"],"F1 score per Tail on Fold {idx} - OpenMax (Panicum dataset)")
TEST_FOLDER = Path("/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum_plantnet/Test/alpha_2/epsilon_0.5/Folds")
VAL_FOLDER = Path("/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum_plantnet/Val/alpha_2/epsilon_0.5/Folds")
BASE_SAVE_DIR = Path("~/Desktop/Plot/OpenMax/").expanduser()
COL_NAME = "f1"

plot_pipeline = PlottingPipeline(TEST_FOLDER,VAL_FOLDER,BASE_SAVE_DIR,COL_NAME,None)
plot_pipeline.run(cfg,show_std=False,format=format)

cfg = PlotConfig("Tail","F1 score",x_ticks,y_ticks,["Test","Validation"],"F1 score mean and standard devation on Test/Val sets per Tail - OpenMax (Panicum dataset)")
TEST_FOLDER = Path("/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum_plantnet/Test/")
VAL_FOLDER = Path("/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum_plantnet/Val/alpha_2/epsilon_0.5/")
BASE_SAVE_DIR = Path("~/Desktop/Plot/OpenMax/").expanduser()
MEAN_COL_NAME = "f1_macro_medio"
STD_COL_NAME = "f1_macro_std"

plot_pipeline = PlottingPipeline(TEST_FOLDER,VAL_FOLDER,BASE_SAVE_DIR,MEAN_COL_NAME,STD_COL_NAME)
plot_pipeline.run(cfg,show_std=True,format=format)
