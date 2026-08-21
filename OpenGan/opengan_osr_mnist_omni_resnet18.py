from Classificador import OSR_classifier,Discriminator
from Feat_extraction.ResNet18_feature_extraction import ResNet18_feature_extraction
from torch.utils.data import DataLoader,ConcatDataset
from Utils_OpenGan import FeatDataset,Matriz_confusao_osr_dataset_outlier
from Utils import *
from sklearn.metrics import accuracy_score,f1_score
import os,gc
import pandas
from Utils import NOMES,metricasImplementadasV2,metricLogger
"""Código para implementar o classificador OSR utilizando OpenGan"""
fix_random_seed(42)
device = "cuda:0"

num_classes = 10
bs=256

# O treino (OpenGAN-IC/opengan_mnist_omni_training.ipynb) grava o discriminador de
# melhor AUROC de validacao em best.DNet, entao nao e mais preciso fixar a epoca.
path_to_D = "/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Mnist_omni/OpenGan_Mnist_omniResNet18/best.DNet"


model_name = NOMES.RESNET18.value
feats_dir = os.path.join(NOMES.FEATS_DIR.value,NOMES.MNIST_OMNI.value,model_name)

result_dir = os.path.join("/home/alexandreselani/Desktop/OpenGan/Resultados/Mnist_omni/",model_name)

# Colunas da matriz de confusao: desconhecida primeiro, depois os digitos do MNIST.
column_names = ["Unknown"] + [str(i) for i in range(num_classes)]
def classificacao(dataloader,osr_classifier):
    all_predicts = []
    all_labels = []
    all_scores = []
    for X,y in dataloader:
        X = X.unsqueeze_(-1).unsqueeze_(-1)
        X = X.to(device)
        predicts,outlier_score = osr_classifier.classify(X)
        all_predicts.append(predicts.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_scores.append(outlier_score.detach().cpu())
    
    all_predicts = torch.cat(all_predicts)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    print(all_scores)
    return all_predicts,all_labels,all_scores

def create_instances(epsilon,num_classes):
    #-------------------------parametros do discriminador-----------------
    #numero de canais
    nc = 512


    # Size of feature maps in discriminator
    # Precisa bater com o valor usado no treino: o best.DNet foi gravado com ndf=64.
    ndf = 128
    discriminator = Discriminator(nc=nc,ndf=ndf).to(device=device)
    discriminator.eval()

    discriminator.load_state_dict(torch.load(path_to_D))

    #---------------------Carregando o classificador
    classifier = ResNet18_feature_extraction(num_classes)
    classifier.load_model(torch.load("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"))
    classifier.model.eval()

    #classificador + discriminador
    osr = OSR_classifier(classifier=classifier,discriminator=discriminator,epsilon=epsilon)

    return discriminator, classifier,osr

def model_selection(epsilons):
    epsilons = [round(epsilon,2) for epsilon in epsilons]

    melhor_f1 = -1
    melhor_epsilon=-1

    #carregamento dos dados de validacao
    mnist_val_data = torch.load(os.path.join(feats_dir, "mnist_val_features.pt"))
    mnist_val_dataset = FeatDataset(mnist_val_data)

    omni_val_data = torch.load(os.path.join(feats_dir, "omni_val_features.pt"))
    omni_val_dataset = FeatDataset(omni_val_data)

    val_dataset = ConcatDataset([omni_val_dataset,mnist_val_dataset])
    val_dataloader = DataLoader(val_dataset,batch_size=bs,shuffle=False)

    val_dir = os.path.join(result_dir,"Val")
    os.makedirs(val_dir,exist_ok=True)

    # n_folds=0: experimento de split unico, sem validacao cruzada.
    metric_logger = metricLogger(epsilons,0,val_dir,mc_column_names=column_names)

    # O discriminador e o classificador nao dependem do epsilon: instancia uma vez
    # e so troca o limiar a cada iteracao.
    discriminator, classifier, osr = create_instances(epsilons[0],num_classes)

    for epsilon in epsilons:
        gc.collect()
        print(f"epsilon {epsilon}")

        osr.set_epsilon(epsilon)

        predicts,labels,outlier_scores = classificacao(val_dataloader,osr)
        metricas = metricasImplementadasV2(predicts,labels,outlier_scores=outlier_scores,metodo="opengan")
        metricas = metricas._metricas()

        f1 = metricas["F1 macro"]

        metric_logger.update(metricas,0,epsilon)
        metric_logger.update_mc(epsilon,predicts,labels,labels)

        if f1>melhor_f1:
            melhor_f1=f1
            melhor_epsilon=epsilon

    metric_logger.aggregate("Val.csv")

    del discriminator,classifier,osr

    print(f"\nMelhor epsilon: {melhor_epsilon} (F1 macro {melhor_f1:.4f})")
   
    
def test_hiperparameters(epsilons):
    epsilons = [round(epsilon,2) for epsilon in epsilons]

    #--------carregamento dos dados de teste---------------------

    mnist_test_data = torch.load(os.path.join(feats_dir, "mnist_test_features.pt"))
    mnist_test_dataset = FeatDataset(mnist_test_data)

    omni_test_data = torch.load(os.path.join(feats_dir, "omni_test_features.pt"))
    omni_test_dataset = FeatDataset(omni_test_data)

    test_dataset = ConcatDataset([omni_test_dataset,mnist_test_dataset])
    test_dataloader = DataLoader(test_dataset,batch_size=bs,shuffle=False)

    test_dir = os.path.join(result_dir,"Test")
    os.makedirs(test_dir,exist_ok=True)

    # n_folds=0: experimento de split unico, sem validacao cruzada.
    metric_logger = metricLogger(epsilons,0,test_dir,mc_column_names=column_names)

    # O discriminador e o classificador nao dependem do epsilon: instancia uma vez
    # e so troca o limiar a cada iteracao.
    discriminator, classifier, osr = create_instances(epsilons[0],num_classes)

    for epsilon in epsilons:
        gc.collect()
        print(f"epsilon {epsilon}")

        osr.set_epsilon(epsilon)

        predicts,labels,outlier_scores = classificacao(test_dataloader,osr)
        metricas = metricasImplementadasV2(predicts,labels,outlier_scores=outlier_scores,metodo="opengan")
        metricas = metricas._metricas()

        metric_logger.update(metricas,0,epsilon)
        metric_logger.update_mc(epsilon,predicts,labels,labels)

    metric_logger.aggregate("Test.csv")

    del discriminator,classifier,osr


if __name__ == "__main__":
    epsilons = np.arange(0.0, 1.0, 0.01).tolist()
    model_selection(epsilons)
    test_hiperparameters(epsilons)

