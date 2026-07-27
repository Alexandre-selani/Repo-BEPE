import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,classification_report

class metricasImplementadasV2:
    def __init__(self,predict=None,label=None,outlier_scores = None,metodo=None):
        self.outlier_scores = outlier_scores
        self.predict = predict
        self.label = label
        self.unknown_class_idx=None
        self.metodo = metodo
        #ajuste das labels para lidar com desconhecidas=-1. Agora, o indice das desconhecidas eh 0
        if metodo is None:
            print("DEFINIR METODO")
            raise ValueError
        elif metodo == "openmax":
            self.unknown_class_idx=0
            self.label= self.label+1
        else:
            self.unknown_class_idx = -1
        
        
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
        return res

    def _accuracy(self) -> float:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))
    
    def _inner_metric(self) -> float:
        """Retorna a acuracia levando em consideracao apenas as amostras de classes CONHECIDAS (Inner metric ou KKC Accuracy)"""
        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,(x,y) in enumerate(zip(self.predict,self.label)) if (y != self.unknown_class_idx and x!= self.unknown_class_idx)] #vetor com os indices das amostras que devem ser verificadas
        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.label[idx]: #se a predicao for correta
                corretas+=1

        if(len(predicoes)>0):
            return float(corretas)/float(len(predicoes))
        
        return 1.0

    def _UUC_Accuracy(self) -> float:
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
            return float(corretas)/float(len(predicoes))
        
        return 1.0
    
    def _outer_metric(self) -> float:
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
        
        return float(corretas)/float(len(self.predict))

    
    def _halfpoint(self) -> float:
        """Uma modificacao do Inner metric que tambem leva em consideracao falsos desconhecidos
        
        """
        assert len(self.predict) == len(self.label)
        
        indices_amostras = [i for i,(y) in enumerate(self.label) if (y != self.unknown_class_idx)] #vetor com os indices das amostras que devem ser verificadas

        predicoes = [self.predict[i] for i in indices_amostras] #amostras a serem consideradas

        

        corretas = 0

        for predicao, idx in zip(predicoes,indices_amostras):
            if predicao == self.label[idx]: #se a predicao for correta
                corretas+=1

        
        return float(corretas)/float(len(predicoes))
    
    def _f1_macro(self) -> float:
        """
        Returns the F1-measure with a macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')
    
    def _AUROC(self) -> float:
        """Retorna a auroc"""
        if self.outlier_scores is None:
            return
        assert len(self.outlier_scores) == len(self.label)
        label = np.array(self.label)
        y_true_bin = (label != self.unknown_class_idx).astype(int)

        if self.metodo == "opengan":
            y_score = self.outlier_scores
        else:
            y_score = 1 - self.outlier_scores
        
        #print(y_score)
        return roc_auc_score(y_true_bin, y_score)

    def per_class_metrics(self):
        return classification_report(y_true=self.label,y_pred=self.predict,output_dict=True,zero_division=0)
