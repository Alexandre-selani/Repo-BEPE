import numpy as np
from sklearn.metrics import f1_score,roc_auc_score

class metricasImplementadas:
    def __init__(self,predict=None,label=None,outlier_scores = None):
        
        self.predict = predict

        #ajuste das labels para lidar com desconhecidas=-1. Agora, o indice das desconhecidas eh 0
        self.unknown_class_idx=0
        self.label = label
        self.label= self.label+1
        self.outlier_scores = outlier_scores
        
    def _metricas(self):
        #print(f"velho {self.label-1} novo {self.label}")
        res = {
        "accuracy": self._accuracy(),
        "inner metric": self._inner_metric(),
        "UUC Accuracy": self._UUC_Accuracy(),
        "outer metric": self._outer_metric(),
        "halfpoint": self._halfpoint(),
        "F1 macro": self._f1_macro(),
        "auroc": self._AUROC()
    }

        # Nova lógica para arredondar 3 casas decimais
        metrics_rounded = {}
        for k, v in res.items():
            if isinstance(v, tuple):
                # Arredonda apenas o primeiro valor (a taxa) e mantém os contadores (inteiros)
                metrics_rounded[k] = (round(v[0], 3), v[1], v[2])
            else:
                # Arredonda o float diretamente (F1 e AUROC)
                metrics_rounded[k] = round(v, 3)
                
        return metrics_rounded
    
    def _accuracy(self) -> tuple[float,int,int]:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict)),correct,len(self.predict)
    
    def _inner_metric(self) -> tuple[float,int,int]:
        """Retorna a acuracia levando em consideracao apenas as amostras de classes CONHECIDAS (Inner metric ou KKC Accuracy)"""
        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,(x,y) in enumerate(zip(self.predict,self.label)) if (y != self.unknown_class_idx and x!= self.unknown_class_idx)] #vetor com os indices das amostras que devem ser verificadas
        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.label[idx]: #se a predicao for correta
                corretas+=1

        if(len(predicoes)>0):
            return float(corretas)/float(len(predicoes)),float(corretas),float(len(predicoes))
        
        return 1, 0, 0

    def _UUC_Accuracy(self) -> tuple[float,int,int]:
        """Retorna a acuracia levando em consideracao apenas as amostras de classes DESCONHECIDAS (UUC Accuracy)
        NAO eh outer metric
        """

        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,y in enumerate(self.label) if y == self.unknown_class_idx] #vetor com os indices das amostras que devem ser verificadas
        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.unknown_class_idx: #se a predicao for correta
                corretas+=1

        if(len(predicoes)>0):
            return float(corretas)/float(len(predicoes)),corretas,len(predicoes)
        
        return 1,0,0
    
    def _outer_metric(self) -> tuple[float,int,int]:
        """Mede a habilidade do classificador de distinguir KKCs e UUCs. Eh um problema de classificacao binaria
        """
        assert len(self.predict) == len(self.label)
        corretas = 0

        for predicao,label_correta in zip(self.predict,self.label):
            if(label_correta == self.unknown_class_idx):#se a amostra for UUC
                if(predicao==self.unknown_class_idx):#se o classificador detectou a novidade
                    corretas+=1
            else:                                   #se a amostra for KKC
                if(predicao!=self.unknown_class_idx): #se a amostra foi classificada como KKC, independente de acertar a classe
                    corretas+=1
        
        return float(corretas)/float(len(self.predict)),corretas,len(self.predict)

    
    def _halfpoint(self) -> tuple[float,int,int]:
        """Uma modificacao do Inner metric que tambem leva em consideracao falsos desconhecidos
        
        """
        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,(x,y) in enumerate(zip(self.predict,self.label)) if (y != self.unknown_class_idx)] #vetor com os indices das amostras que devem ser verificadas

        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.label[idx]: #se a predicao for correta
                corretas+=1

        
        return float(corretas)/float(len(predicoes)),corretas,len(predicoes)
    
    def _f1_macro(self) -> float:
        """
        Returns the F1-measure with a macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')
    
    def _AUROC(self) -> float:
        """Retorna a auroc"""
        assert self.outlier_scores is not None
        assert len(self.outlier_scores) == len(self.label)
        y_true_bin = (self.label != self.unknown_class_idx).astype(int)
        y_score = 1 - self.outlier_scores
        return roc_auc_score(y_true_bin, y_score)


        