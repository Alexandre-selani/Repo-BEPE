
import matplotlib.pyplot as plt
from Utils import metricasImplementadas
class AnaliseGraficaTest:

    def __init__(self,nome:str,nome_dataset:str, dir:str=None):
        self.nome=nome
        self.nome_dataset=nome_dataset
        self.test_accuracy=[]
        self.inner_metric=[]
        self.outer_metric=[]
        self.halfpoint=[]
        self.uuc_accuracy=[]
        self.F1=[]
        self.epochs=[]
        
        if dir:
            self.dir = dir
        else:
            self.dir = "/home/alexandreselani/Desktop/OpenGan/Resultados"

        
        
        self.titulo = f"metricas do {self.nome} "

    def addEpochTest(self,metricas:metricasImplementadas=None,epoch:int=None):
        self.epochs.append(epoch)

        if(metricas):
            self.test_accuracy.append(metricas["accuracy"][0])
            self.inner_metric.append(metricas["inner metric"][0])
            self.outer_metric.append(metricas["outer metric"][0])
            self.halfpoint.append(metricas["halfpoint"][0])
            self.uuc_accuracy.append(metricas["UUC Accuracy"][0])
            self.F1.append(metricas["F1 macro"])
        else:
            raise ValueError     

    def mostraGraficoTest(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.epochs, self.test_accuracy, color='red', label='Acurácia')
        plt.plot(self.epochs, self.inner_metric, color='blue', label='Inner metric')
        plt.plot(self.epochs, self.outer_metric, color='orange', label='Outer metric')
        plt.plot(self.epochs, self.halfpoint, color='green', label='Halfpoint')
        plt.plot(self.epochs, self.uuc_accuracy, color='black', label='UUC Accuracy')
        #
        plt.plot(self.epochs, self.F1, color='purple', label='F1 Macro')
        
        
            
        plt.title(self.titulo)

        plt.xlabel("Épocas")
        plt.xticks(self.epochs)
        plt.ylabel("Valor da Métrica")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(self.dir + f"metricas_{self.nome}_{self.nome_dataset}.png")
        plt.close()

   

        