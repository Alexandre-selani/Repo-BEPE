import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
device = "cuda:0"

def ToUnknown():
    """Retorna uma função que mapeia qualquer target para -1 (classe desconhecida).

    Útil para definir o comportamento de rejeição em problemas de Open Set Recognition.

    Returns:
        function: Função lambda que recebe um target e retorna -1.
    """
    return lambda target: -1

def hadamardProduct(features,classWeights,targets):
    """Calcula o produto de Hadamard (multiplicação elemento a elemento) entre
    as features e os pesos da classe correspondente.

    Args:
        features (torch.Tensor): Tensor de características extraídas do modelo.
        classWeights (torch.Tensor): Pesos por classe do modelo.
        targets (torch.Tensor): Rótulos das amostras.

    Returns:
        torch.Tensor: Resultado da multiplicação elemento a elemento.
    """
    #element-wise multiplication
    with torch.no_grad():
        return torch.mul(features,classWeights[targets])

def concatFeatures(preAttenuated,hadamard):
    """Concatena as features pré-atenuadas com o resultado do produto de Hadamard.

    Args:
        preAttenuated (torch.Tensor): Features originais antes da atenuação.
        hadamard (torch.Tensor): Resultado do produto de Hadamard.

    Returns:
        torch.Tensor: Tensor concatenado ao longo da dimensão 1 (features).
    """
    with torch.no_grad():
        return torch.concat((preAttenuated,hadamard),dim=1)

def calculateMeanConcatenatedVectors(concatVectors,targets,num_classes):
    """Calcula a média dos vetores concatenados para cada classe.

    Para cada classe única presente nos targets, calcula a média dos vetores
    concatenados pertencentes àquela classe.

    Args:
        concatVectors (torch.Tensor): Tensor de vetores concatenados.
        targets (torch.Tensor): Rótulos das amostras.

    Returns:
        torch.Tensor: Tensor empilhado com os vetores médios de cada classe.
    """
    feature_dim = concatVectors.shape[1]
    # Inicializa com zeros para todas as classes possíveis
    class_means = torch.zeros((num_classes, feature_dim), dtype=concatVectors.dtype)

    for c in range(num_classes):
        mask = targets == c
        if torch.any(mask):
            with torch.no_grad():
                class_means[c] = concatVectors[mask].mean(dim=0)

    return class_means

def GNL(ltmin,ltmax,logits):
    """Global Normalized Logits — Normaliza os logits para o intervalo [0, 1].

    Aplica uma normalização min-max nos logits usando os limites mínimo e máximo
    observados durante o treinamento. Caso a faixa seja zero, retorna 9999 como
    valor de fallback para indicar indefinição.

    Args:
        ltmin (float): Valor mínimo do logit observado no treinamento.
        ltmax (float): Valor máximo do logit observado no treinamento.
        logits (torch.Tensor): Tensor de logits a ser normalizado.

    Returns:
        torch.Tensor: Logits normalizados no intervalo [0, 1].
    """
    div = (ltmax - ltmin)
    if div == 0:
        norm = 9999
    else:
        norm = (logits - ltmin) / div 
    with torch.no_grad():
        return torch.clamp(norm, 0, 1)

def calculateMagnitude(vectors):
    """Calcula a norma L2 (magnitude) dos vetores ao longo da dimensão 1.

    Args:
        vectors (torch.Tensor): Tensor de vetores de entrada.

    Returns:
        torch.Tensor: Tensor com as normas L2 de cada vetor.
    """
    return torch.linalg.vector_norm(vectors,dim=1)

def costarrSimilarity(magnitudesMeans,magnitudesConcatenatedVectors,means,concatenatedVectors,max_logits_idx):
    """Calcula a similaridade baseada no cosseno entre os vetores concatenados
    e as médias das classes, adaptada para o método COSTARR.

    A similaridade é calculada como:
        0.5 * (1 + cosseno(vetor, media_da_classe))

    Args:
        magnitudesMeans (torch.Tensor): Magnitudes dos vetores médios de cada classe.
        magnitudesConcatenatedVectors (torch.Tensor): Magnitudes dos vetores concatenados.
        means (torch.Tensor): Vetores médios de cada classe.
        concatenatedVectors (torch.Tensor): Vetores concatenados das amostras.
        max_logits_idx (torch.Tensor): Índices das classes com maior logit.

    Returns:
        torch.Tensor: Valores de similaridade no intervalo [0, 1].
    """
    similarity = 1 + (torch.sum(concatenatedVectors*means[max_logits_idx],dim=1)/torch.mul(magnitudesMeans[max_logits_idx],magnitudesConcatenatedVectors))
    
    return 0.5 * similarity

def costarrPredict(model,testloader,train_calc):
    """Realiza predições usando o método COSTARR para Open Set Recognition.

    Processa todo o test loader, calculando:
    1. Logits normalizados via GNL
    2. Produto de Hadamard entre features e pesos das classes
    3. Similaridade entre vetores concatenados e médias das classes
    4. Score final = logits_normalizados * similaridade

    Args:
        model (nn.Module): Modelo treinado com pesos por classe.
        testloader (DataLoader): DataLoader com os dados de teste.
        train_calc (dict): Dicionário com 'min_logit', 'max_logit' e 'means'
                           calculados na etapa de fit.
        threshold (float): Limiar para decisão (não utilizado internamente,
                           reservado para interface).

    Returns:
        tuple: (score, max_logits, max_logits_idx, all_targets)
            - score (torch.Tensor): Score final COSTARR para cada amostra.
            - max_logits (torch.Tensor): Valor do maior logit por amostra.
            - max_logits_idx (torch.Tensor): Classe predita (índice do maior logit).
            - all_targets (torch.Tensor): Rótulos verdadeiros.
    """
    all_logits = []
    all_features = []
    all_targets = []
    
    min_train_logit = train_calc["min_logit"]
    max_train_logit = train_calc["max_logit"]
    meanPerClassVector = train_calc["means"]
    print(meanPerClassVector)
    for i,(X, y) in enumerate(testloader):
        
        X = X.to(device)
        with torch.no_grad():
            logits,features = model(X)

        all_logits.append(logits.cpu().detach())
        all_features.append(features.cpu().detach())
        all_targets.append(y.detach())
    
    all_logits = torch.cat(all_logits)
    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)

    max_logits,max_logits_idx = torch.max(all_logits,dim=1)

    normalized_logits = GNL(min_train_logit,max_train_logit,max_logits)
    print(normalized_logits)
    hadamard = hadamardProduct(all_features,model.getPerClassWeights().cpu(),max_logits_idx)
    concatenatedVectors = concatFeatures(all_features,hadamard)

    magnitudesMeans = calculateMagnitude(meanPerClassVector)
    magnitudesConcatenatedVectors = calculateMagnitude(concatenatedVectors)

    score = normalized_logits * costarrSimilarity(magnitudesMeans,magnitudesConcatenatedVectors,meanPerClassVector,concatenatedVectors,max_logits_idx)

    return score,max_logits,max_logits_idx,all_targets

def thresholdPredicitions(score,max_logits_idx,epsilon):
    """Aplica um limiar (threshold) nas predições para Open Set Recognition.

    Amostras com score abaixo do limiar epsilon são classificadas como
    desconhecidas (-1). Caso contrário, mantém a classe predita original.

    Args:
        score (torch.Tensor): Score COSTARR para cada amostra.
        max_logits_idx (torch.Tensor): Índices das classes com maior logit.
        epsilon (float): Limiar de decisão.

    Returns:
        torch.Tensor: Predições finais, onde -1 indica classe desconhecida.
    """
    predictions = torch.where(score < epsilon, -1, max_logits_idx)

    return predictions

def costarrFit(model,trainloader,save_dir):
    """Calcula os parâmetros de treinamento necessários para o método COSTARR.

    Processa todo o train loader, considerando apenas amostras classificadas
    corretamente, e calcula:
    1. Logits mínimo e máximo (para normalização GNL)
    2. Vetores concatenados médios por classe (produto de Hadamard + concatenação)
    3. Salva os parâmetros em um arquivo .pt

    Args:
        model (nn.Module): Modelo treinado com pesos por classe.
        trainloader (DataLoader): DataLoader com os dados de treinamento.
        save_dir (str): Caminho (sem extensão) para salvar o dicionário .pt.

    Returns:
        None: O dicionário com 'means', 'min_logit' e 'max_logit' é salvo em
              disco no arquivo <save_dir>
    """
    all_logits = []
    all_features = []
    all_targets = []

    for i,(X, y) in enumerate(trainloader):
        
        X = X.to(device)
        with torch.no_grad():
            logits,features = model(X)

        all_logits.append(logits.cpu().detach())
        all_features.append(features.cpu().detach())
        all_targets.append(y.detach())
    
    all_logits = torch.cat(all_logits)
    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)

    correctly_classified = torch.argmax(all_logits,1)==all_targets
    
    correct_logits = all_logits[correctly_classified]
    correct_features = all_features[correctly_classified]
    correct_targets = all_targets[correctly_classified]
    
    # CORREÇÃO: Mapeia apenas os maiores logits das predições corretas
    max_correct_logits = torch.max(correct_logits, dim=1)[0]
    
    # CORREÇÃO: Min e Max baseados estritamente nos max_logits (sem desempacotamento inválido)
    min_logit = torch.min(max_correct_logits)
    max_logit = torch.max(max_correct_logits)
    
    hadamard = hadamardProduct(correct_features,model.getPerClassWeights().cpu(),correct_targets)
    concatenatedVectors = concatFeatures(correct_features,hadamard)
    meanConcatenatedVectors = calculateMeanConcatenatedVectors(concatenatedVectors,correct_targets,len(np.unique(all_targets)))

    costarr = {"means":meanConcatenatedVectors,
               "min_logit":min_logit,
               "max_logit":max_logit}
    
    torch.save(costarr,save_dir)

def scorePorClasse(scores, labels):
    """Descreve a média e o desvio padrão dos scores por classe.

    Args:
        scores (torch.Tensor ou np.ndarray): Scores de cada amostra.
        labels (torch.Tensor ou np.ndarray): Rótulos verdadeiros de cada amostra.

    Returns:
        dict: Dicionário no formato {classe: {'media': float, 'std': float, 'n': int}}.
              A classe -1 representa as amostras desconhecidas (UUC).
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    classes_unicas = np.unique(labels)
    resultado = {}

    for classe in sorted(classes_unicas, key=lambda c: (c == -1, c)):
        mask = labels == classe
        scores_classe = scores[mask]

        nome = "Desconhecida (-1)" if classe == -1 else f"Classe {classe}"
        resultado[nome] = {
            "media": float(np.mean(scores_classe)),
            "std": float(np.std(scores_classe)),
            "n": int(np.sum(mask)),
        }

    # Resumo geral
    resultado["___GERAL___"] = {
        "media": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n": len(scores),
    }

    return resultado
