import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
device = "cuda:0"

def ToUnknown():
    return lambda target: -1

def hadamardProduct(features,classWeights,targets):
    #element-wise multiplication
    with torch.no_grad():
        return torch.mul(features,classWeights[targets])

def concatFeatures(preAttenuated,hadamard):
    with torch.no_grad():
        return torch.concat((preAttenuated,hadamard),dim=1)

def calculateMeanConcatenatedVectors(concatVectors,targets):
    unique_classes = torch.unique(targets)
    class_means = []
    for c in unique_classes:
        mask = targets == c
        class_vectors = concatVectors[mask]
        with torch.no_grad():
            class_mean = class_vectors.mean(dim=0)
        class_means.append(class_mean)
    return torch.stack(class_means)

def GNL(ltmin,ltmax,logits):
    div = (ltmax - ltmin)
    if div == 0:
        norm = 9999
    else:
        norm = (logits - ltmin) / div 
    with torch.no_grad():
        return torch.clamp(norm, 0, 1)

def calculateMagnitude(vectors):
    return torch.linalg.vector_norm(vectors,dim=1)

def costarrSimilarity(magnitudesMeans,magnitudesConcatenatedVectors,means,concatenatedVectors,max_logits_idx):
    similarity = 1 + (torch.sum(concatenatedVectors*means[max_logits_idx],dim=1)/torch.mul(magnitudesMeans[max_logits_idx],magnitudesConcatenatedVectors))

    return 0.5 * similarity

def costarrPredict(model,testloader,train_calc):
    all_logits = []
    all_features = []
    all_targets = []
    
    min_train_logit = train_calc["min_logit"]
    max_train_logit = train_calc["max_logit"]
    meanPerClassVector = train_calc["means"]
    
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

    hadamard = hadamardProduct(all_features,model.getPerClassWeights().cpu(),max_logits_idx)
    concatenatedVectors = concatFeatures(all_features,hadamard)

    magnitudesMeans = calculateMagnitude(meanPerClassVector)
    magnitudesConcatenatedVectors = calculateMagnitude(concatenatedVectors)

    score = normalized_logits *costarrSimilarity(magnitudesMeans,magnitudesConcatenatedVectors,meanPerClassVector,concatenatedVectors,max_logits_idx)

    # Predicao = argmax, exceto onde score < 0.5 → desconhecido (-1)
    predictions = torch.where(score < 0.2, -1, max_logits_idx)

    return predictions,all_targets

def costarrFit(model,trainloader,save_dir):
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
    meanConcatenatedVectors = calculateMeanConcatenatedVectors(concatenatedVectors,correct_targets)

    costarr = {"means":meanConcatenatedVectors,
               "min_logit":min_logit,
               "max_logit":max_logit}
    
    torch.save(costarr,save_dir+".pt")
