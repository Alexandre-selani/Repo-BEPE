
import matplotlib.pyplot as plt
from . import metricasImplementadas
class AnaliseGrafica:

    def __init__(self, nome:str,nome_dataset:str):
        self.nome=nome
        self.nome_dataset=nome_dataset
        self.test_accuracy=[]
        self.inner_metric=[]
        self.outer_metric=[]
        self.halfpoint=[]
        self.uuc_accuracy=[]
        self.F1=[]
        self.epochs=[]
        self.dir="/home/alexandreselani/Desktop/pytorch-ood/resultados/"
        self.train_loss=[]
        self.train_acc=[]

        self.val_loss=[]
        self.val_acc=[]
        
        self.titulo = f"metricas do {self.nome} "

    def addEpoch(self,metricas:metricasImplementadas=None,epoch:int=None,train_loss=None,train_acc=None,val_loss=None,val_acc=None):
        self.epochs.append(epoch)

        if(metricas):
            self.test_accuracy.append(metricas["accuracy"][0])
            self.inner_metric.append(metricas["inner metric"][0])
            self.outer_metric.append(metricas["outer metric"][0])
            self.halfpoint.append(metricas["halfpoint"][0])
            self.uuc_accuracy.append(metricas["UUC Accuracy"][0])
            self.F1.append(metricas["F1 macro"])

        if train_loss:
            self.train_loss.append(train_loss)
        if train_acc:
            self.train_acc.append(train_acc)
        if val_loss:
            self.val_loss.append(val_loss)
        if val_acc:
            self.val_acc.append(val_acc)

        # print(f"{self.nome} inner metric is %.3f ({metricas.certas_inner}/{metricas.total_inner})" % (metricas.inner_metric))
        # print(f"{self.nome} outer metric is %.3f ({metricas.certas_outer}/{metricas.total_outer})" % (metricas.outer_metric))
        # print(f"{self.nome} halfpoint is %.3f ({metricas.certas_halfpoint}/{metricas.total_halfpoint})" % (metricas.halfpoint))
        # print(f"{self.nome} uuc accuracy is %.3f ({metricas.certas_uuc_accuracy}/{metricas.total_ucc_accuracy})" % (metricas.uuc_accuracy))
        # print(f"{self.nome} accuracy is %.3f" % (metricas.accuracy))


        #print(f"{self.nome} F1 is %.3f" % (metricas.f1_measure))
        #print(f"{self.nome} f1_macro is %.3f" % (metricas.f1_macro))
        #print(f"{self.nome} f1_macro_weighted is %.3f" % (metricas.f1_macro_weighted))
        #print(f"{self.nome} area_under_roc is %.3f" % (metricas.area_under_roc))
        print(f"_________________________________________")
    
    def mostraGrafico(self):
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

        if((self.train_loss and self.train_acc and self.val_acc and self.val_loss)):
            plt.figure(figsize=(12, 8))
            plt.plot(self.epochs, self.val_acc, color='red', label='Acurácia na validacao')
            plt.plot(self.epochs, self.val_loss, color='purple', label='Erro na validacao')
            plt.plot(self.epochs, self.train_loss, color='blue', label='Erro no treino')
            plt.plot(self.epochs, self.train_acc, color='orange', label='Acuracia no treino')
            
            
                
            plt.title("Curva de erro (treino/validacao)")

            plt.xlabel("Épocas")
            plt.xticks(self.epochs)
            plt.ylabel("Valor da Métrica")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.savefig(self.dir + f"curva_de_erro{self.nome}_{self.nome_dataset}.png")


        
        #plt.show()

        