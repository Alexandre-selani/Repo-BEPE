
import matplotlib.pyplot as plt
import os
class AnaliseGraficaVal:

    def __init__(self,nome:str,nome_dataset:str, dir:str=None):
        self.nome=nome
        self.nome_dataset=nome_dataset
        self.epochs=[]
        
        if dir:
            self.dir = dir
        else:
            self.dir = "/home/alexandreselani/Desktop/OpenGan/Resultados/"
            
        self.train_loss=[]
        self.train_acc=[]

        self.val_loss=[]
        self.val_acc=[]
        
        self.titulo = f"curvas validacao do {self.nome} "

    
        

    def addEpochVal(self,epoch:int=None,train_loss=None,train_acc=None,val_loss=None,val_acc=None):
        
        self.epochs.append(epoch)

        if train_loss:
            self.train_loss.append(train_loss)
        if train_acc:
            self.train_acc.append(train_acc)
        if val_loss:
            self.val_loss.append(val_loss)
        if val_acc:
            self.val_acc.append(val_acc)

    def mostraGraficoVal(self):
        if((self.train_loss and self.train_acc and self.val_acc and self.val_loss)):
            plt.figure(figsize=(12, 8))
            plt.plot(self.epochs, self.val_acc, color='red', label='Acurácia na validação')
            plt.plot(self.epochs, self.val_loss, color='purple', label='Erro na validação')
            plt.plot(self.epochs, self.train_loss, color='blue', label='Erro no treino')
            plt.plot(self.epochs, self.train_acc, color='orange', label='Acurácia no treino')
            
            
                
            plt.title("Curva de erro (treino/validação)")

            plt.xlabel("Épocas")
            plt.xticks(self.epochs,rotation=45)
            plt.ylabel("Valor da Métrica")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.savefig(os.path.join(self.dir,f"curva_de_erro_{self.nome}_{self.nome_dataset}.png"))
            plt.close()

    