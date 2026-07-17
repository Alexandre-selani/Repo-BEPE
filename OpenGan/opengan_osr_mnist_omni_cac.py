from Classificador import OSR_classifier,Discriminator
from Feat_extraction.LeNet_feature_extraction import LeNet_cac_feature_extraction
from torch.utils.data import DataLoader,ConcatDataset
from Utils_OpenGan import FeatDataset,Matriz_confusao_osr_dataset_outlier
from Utils import *
from sklearn.metrics import accuracy_score,f1_score
import os,gc
import pandas
import matplotlib

from Utils import NOMES
"""Código para implementar o classificador OSR utilizando OpenGan"""
fix_random_seed(42)
device = "cuda:0"

num_classes = 10
best_epoch = 140 #constatado previamente por meio de uma selecao de modelo do discriminador
bs=256


model_name = NOMES.LENET.value
feats_dir = "/home/alexandreselani/Desktop/Features_extraidas/Mnist_Omni/LeNet/CAC"

result_dir = os.path.join("/home/alexandreselani/Desktop/OpenGan/Resultados/Mnist_omni_cac/",model_name)
os.makedirs(result_dir, exist_ok=True)
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
    
    for k,j in zip(all_labels,all_predicts):
        if(k >=0):
            print(k, j)
    return all_predicts,all_labels,all_scores

def create_instances(epsilon,num_classes,best_epoch):
    #-------------------------parametros do discriminador-----------------
    #numero de canais
    nc = 84


    # Size of feature maps in discriminator
    ndf = 100
    discriminator = Discriminator(nc=nc,ndf=ndf).to(device=device)
    discriminator.eval()
        
    path_to_D = os.path.join("/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Mnist_omni_CAC/OpenGan_Mnist_omniLeNet_CAC",'epoch-{}.DNet'.format(best_epoch))

    discriminator.load_state_dict(torch.load(path_to_D))

    #---------------------Carregando o classificador
    classifier = LeNet_cac_feature_extraction(num_classes)
    classifier.load_model(torch.load("/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet_cac/LeNet_mnist_omni_cac.pt"))
    classifier.model.eval()

    #classificador + discriminador
    osr = OSR_classifier(classifier=classifier,discriminator=discriminator,epsilon=epsilon)

    return discriminator, classifier,osr

def model_selection(epsilons):
    melhor_f1 = -1
    melhor_epsilon=-1

    #carregamento dos dados de validacao
    mnist_val_data = torch.load(os.path.join(feats_dir, "mnist_val_features.pt"))
    mnist_val_dataset = FeatDataset(mnist_val_data)

    omni_val_data = torch.load(os.path.join(feats_dir, "omni_val_features.pt"))
    omni_val_dataset = FeatDataset(omni_val_data)
    
    val_dataset = ConcatDataset([omni_val_dataset,mnist_val_dataset])
    val_dataloader = DataLoader(val_dataset,batch_size=bs,shuffle=False)
    
    results_by_epsilon = {}

    for epsilon in epsilons:
        epsilon = round(epsilon,2)
        gc.collect()

        discriminator, classifier, osr = create_instances(epsilon,num_classes,best_epoch)
        predicts,labels,outlier_scores = classificacao(val_dataloader,osr)
        metricas = metricasImplementadas(predicts,labels,outlier_scores=outlier_scores,metodo="opengan")
        metricas = metricas._metricas()
        
        f1 = metricas["F1 macro"]

        results_by_epsilon[epsilon] = {
                "epsilon": epsilon,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]
                }
        
        
        if f1>melhor_f1:
            melhor_f1=f1
            melhor_epsilon=epsilon

        del discriminator,classifier,osr,metricas

    final_data = []

    for epsilon in sorted(results_by_epsilon.keys()):
        metrics = results_by_epsilon[epsilon]
        final_data.append(metrics)

    df = pandas.DataFrame(final_data)

    os.makedirs(name=result_dir,exist_ok=True)
    df.to_csv(os.path.join(result_dir,"Resultados_model_selection.csv"),index=False,float_format="%.3f")
   
    
def main():
    
    epsilon = 0.07

    #--------carregamento dos dados de teste---------------------

    mnist_test_data = torch.load(os.path.join(feats_dir, "mnist_test_features.pt"))
    mnist_test_dataset = FeatDataset(mnist_test_data)

    omni_test_data = torch.load(os.path.join(feats_dir, "omni_test_features.pt"))
    omni_test_dataset = FeatDataset(omni_test_data)
    
    test_dataset = ConcatDataset([omni_test_dataset,mnist_test_dataset])
    test_dataloader = DataLoader(test_dataset,batch_size=bs,shuffle=False)
    

    discriminator, classifier, osr = create_instances(epsilon,num_classes,best_epoch)
    predicts,labels,outlier_scores = classificacao(test_dataloader,osr)

    metricas = metricasImplementadas(predicts,labels,outlier_scores=outlier_scores,metodo="opengan")
    metricas = metricas._metricas()
    mc = Matriz_confusao_osr_dataset_outlier(predicts,labels,labels,[],col_labels=["Unknown","0","1","2","3","4","5","6","7","8","9"])
    mc.computa_matriz()
    mc.exibe_matriz(dir=os.path.join("/home/alexandreselani/Desktop/OpenGan/Resultados/Mnist_omni_cac",model_name,f"Mnist_omniglot_eps_{epsilon}"))
    

    results_by_epsilon ={}
    results_by_epsilon[epsilon] = {
                "epsilon": epsilon,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]
                }

    final_data = []

    for epsilon in sorted(results_by_epsilon.keys()):
        metrics = results_by_epsilon[epsilon]
        final_data.append(metrics)

    df = pandas.DataFrame(final_data)

    os.makedirs(name=result_dir,exist_ok=True)
    df.to_csv(os.path.join(result_dir,f"Resultados_test_eps_{epsilon}.csv"),index=False,float_format="%.3f")
    


if __name__ == "__main__":
    epsilons = np.arange(0.0, 0.3, 0.01).tolist()
    #model_selection(epsilons)
    main()

