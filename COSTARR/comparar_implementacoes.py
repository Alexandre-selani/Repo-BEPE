"""
Compara as duas implementações do método COSTARR:
    - COSTARR.py  (versão orientada a objetos, estilo "postprocessor")
    - funcs.py    (versão funcional)

O QUE O SCRIPT FAZ
-------------------
  1) Tenta instanciar a classe COSTARR exatamente como está em COSTARR.py
     e mostra o erro real que ela produz hoje (setup() está quebrado).
  2) Aplica uma correção MÍNIMA e SÓ EM MEMÓRIA (não mexe no seu arquivo em
     disco) para deixar a classe executável.
  3) Calcula o mesmo score usando as funções de funcs.py.
  4) Compara os dois resultados número a número.

COMO USAR
---------
  - Coloque este arquivo na MESMA PASTA que COSTARR.py e funcs.py.
  - Por padrão usa dados sintéticos (reprodutíveis via seed), então roda em
    qualquer máquina, sem precisar do seu modelo/dataloader reais.
  - Se quiser testar com dados reais, veja a função `carregar_dados()` mais
    abaixo — é só substituir o conteúdo dela.
  - Rode:  python3 comparar_costarr.py

DEPENDÊNCIAS: torch, numpy, tqdm, scipy (as mesmas que COSTARR.py/funcs.py já usam)
"""

import os
import sys
import traceback
from Modelos import ResNet18Featurizer
from Datasets import Mnist_omni_loader
from Utils import NOMES
import torch

# --------------------------------------------------------------------------
# 0) CONFIGURAÇÃO
# --------------------------------------------------------------------------
PASTA_DOS_ARQUIVOS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA_DOS_ARQUIVOS)

SEED = 0
N_CLASSES=10
# N_AMOSTRAS = 200      # nº de amostras sintéticas
# DIM_FEATURES = 5      # dimensão do vetor de features
# N_CLASSES = 10         # nº de classes

# test_logits = torch.randn(N_AMOSTRAS, N_CLASSES)
# test_features = torch.randn(N_AMOSTRAS, DIM_FEATURES)
torch.manual_seed(SEED)

try:
    import COSTARR_official as C
    import funcs as F
except ModuleNotFoundError as e:
    print(f"ERRO ao importar: {e}")
    print("Verifique se este script está na mesma pasta que COSTARR.py e funcs.py.")
    sys.exit(1)


device = "cuda:0"

def carregar_dados(modelo, dataloader):
    """Extrai logits, features e pesos de um modelo + dataloader.

    Processa todo o dataloader com o modelo, coletando:
      - logits  → tensor (N, K)
      - features → tensor (N, D)
      - weights  → tensor (K, D) via modelo.getPerClassWeights()

    Args:
        modelo (nn.Module): Modelo treinado que retorna (logits, features)
                            no forward e possui getPerClassWeights().
        dataloader (DataLoader): DataLoader com (X, y).

    Returns:
        tuple: (logits, FVs, weights)
            - logits (torch.Tensor): (N, K)
            - FVs (torch.Tensor):    (N, D)
            - weights (torch.Tensor): (K, D)
    """
    all_logits = []
    all_features = []

    for X, _ in dataloader:
        X = X.to(device)
        with torch.no_grad():
            logits, features = modelo(X)
        all_logits.append(logits.cpu().detach())
        all_features.append(features.cpu().detach())

    logits = torch.cat(all_logits,dim=0)
    FVs = torch.cat(all_features,dim=0)
    weights = modelo.getPerClassWeights().cpu()

    return logits, FVs, weights


def passo1_costarr_original(logits, FVs, weights):
    print("\n" + "=" * 72)
    print("PASSO 1 — instanciando COSTARR exatamente como está em COSTARR.py")
    print("=" * 72)
    try:
        C.COSTARR(logits, FVs, weights)
        print("-> Instanciou sem erro (isso seria inesperado com o código atual).")
    except Exception:
        print("-> ERRO (esperado). setup() referencia 'std' e 'hstd', que nunca")
        print("   são definidos dentro da função:\n")
        traceback.print_exc()


def passo2_costarr_corrigido(logits, FVs, weights):
    print("\n" + "=" * 72)
    print("PASSO 2 — corrigindo setup() (só em memória) e rodando de novo")
    print("=" * 72)
    print("Troca aplicada: em vez de")
    print("    class_models[c] = std, mean, hmean, hstd   # std/hstd indefinidos")
    print("guardamos apenas")
    print("    class_models[c] = mean, hmean               # o que norm() espera")
    print("Isso NÃO altera o COSTARR.py no seu disco — é um patch temporário")
    print("só para conseguirmos rodar a lógica e comparar.\n")

    def setup_corrigido(self, logits, FV):
        preds = torch.max(logits, dim=1).indices
        classes = torch.unique(preds).long().tolist()
        class_models = {}
        self.logit_min = torch.min(logits)
        self.logit_max = torch.max(logits)
        for c in classes:
            select_class_FVs = FV[preds == c]
            hmean = torch.mean(select_class_FVs * self.weights[c], dim=0)
            mean = torch.mean(select_class_FVs, dim=0)
            class_models[c] = mean, hmean
        return class_models

    C.COSTARR.setup = setup_corrigido  # patch só na classe em memória

    modelo = C.COSTARR(logits, FVs, weights)
    scores_matriz = modelo.ReScore(logits, FVs)  # (N, K), só a coluna da classe prevista é != 0
    pred = torch.max(logits, dim=1).indices
    scores = scores_matriz[torch.arange(logits.shape[0]), pred]
    print("-> OK. Scores calculados com a lógica de COSTARR.py (corrigida).")
    return scores, pred, modelo


def passo3_funcs(logits, FVs, weights, pred):
    print("\n" + "=" * 72)
    print("PASSO 3 — calculando o mesmo score com as funções de funcs.py")
    print("=" * 72)

    hadamard = F.hadamardProduct(FVs, weights, pred)
    vetores_concat = F.concatFeatures(FVs, hadamard)
    medias = F.calculateMeanConcatenatedVectors(vetores_concat, pred, N_CLASSES)

    logit_min = torch.min(logits)
    logit_max = torch.max(logits)
    
    return medias,logit_min,logit_max


def passo4_funcs_test(medias,logit_min, logit_max, test_logits,test_features, weights, test_preds):
    print("\n" + "=" * 72)
    print("PASSO 3 — calculando o mesmo score com as funções de funcs.py")
    print("=" * 72)

    hadamard = F.hadamardProduct(test_features, weights, test_preds)
    vetores_concat = F.concatFeatures(test_features, hadamard)
    
    max_logits, max_logits_idx = torch.max(test_logits, dim=1)
    logits_normalizados = F.GNL(logit_min, logit_max, max_logits)

    mag_medias = F.calculateMagnitude(medias)
    mag_concat = F.calculateMagnitude(vetores_concat)
    similaridade = F.costarrSimilarity(mag_medias, mag_concat, medias, vetores_concat, max_logits_idx)

    scores = logits_normalizados * similaridade
    print("-> OK. Scores calculados com funcs.py.")
    return scores

def passo5_costarr_test(modelo,test_features,test_logits):
    
    scores_matriz = modelo.ReScore(test_logits,test_features)  # (N, K), só a coluna da classe prevista é != 0
    pred = torch.max(test_logits, dim=1).indices
    scores = scores_matriz[torch.arange(test_logits.shape[0]), pred]
    print("-> OK. Scores calculados com a lógica de COSTARR.py (corrigida).")
    return scores, pred

def passo6_comparar(scores_costarr, scores_funcs):
    print("\n" + "=" * 72)
    print("PASSO 4 — comparando os dois resultados")
    print("=" * 72)

    diff = (scores_costarr - scores_funcs).abs()
    print(f"Diferença máxima absoluta : {diff.max().item():.2e}")
    print(f"Diferença média absoluta  : {diff.mean().item():.2e}")
    print(f"torch.allclose(atol=1e-5) : {torch.allclose(scores_costarr, scores_funcs, atol=1e-5)}")

    n_mostrar = min(50, len(scores_costarr))
    print(f"\nPrimeiras {n_mostrar} amostras:")
    print(f"{'amostra':>8} | {'COSTARR.py':>12} | {'funcs.py':>12} | {'diferença':>10}")
    print("-" * 50)
    for i in range(n_mostrar):
        print(f"{i:>8} | {scores_costarr[i].item():12.6f} | {scores_funcs[i].item():12.6f} | {diff[i].item():10.2e}")


def resumo_final():
    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)
    print("""
Se a "diferença máxima absoluta" no passo 4 ficou na ordem de 1e-6 ou
menor: as duas implementações calculam exatamente a mesma fórmula
(mesma matemática, mesmo resultado). A diferença real entre os arquivos,
hoje, é:

  1. COSTARR.py não roda do jeito que está salvo (bug em setup(), ver
     passo 1) — precisa da correção do passo 2 para funcionar.
  2. Depois de corrigido, ainda restam diferenças de implementação (não
     de lógica):
       - funcs.py filtra "amostras corretamente classificadas" e agrupa
         por classe verdadeira; COSTARR.py assume que quem chama a classe
         já filtrou isso e agrupa pela classe PREVISTA.
       - funcs.py é vetorizado (mais rápido); COSTARR.py itera classe a
         classe com um for + tqdm.
       - COSTARR.py usa torch.nn.functional.cosine_similarity (tem um
         epsilon de segurança); funcs.py calcula manualmente (sem esse
         epsilon).
       - funcs.py trata divisão por zero em GNL (quando max==min dos
         logits); COSTARR.py não trata esse caso.
""")


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Se quiser usar dados reais de um modelo + dataloader, descomente:
    #
    #   from meu_modulo import MeuModelo, get_dataloader
    #   modelo = MeuModelo().to(device)
    #   modelo.load_state_dict(torch.load("checkpoint.pt"))
    #   modelo.eval()
    #   dataloader = get_dataloader()
    #   logits, FVs, weights = carregar_dados(modelo, dataloader)
    #
    # Por padrão usa dados sintéticos (sem dependência externa):
    # -------------------------------------------------------------------
    featurizer = ResNet18Featurizer().to(device)
    featurizer.eval()
    featurizer.load_state_dict(torch.load("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"))
    
    data = Mnist_omni_loader(bs=256,transform=NOMES.RESNET18_MNIST_OMNI_TRANSFORMS.value)
    train_loader = data.load_train()
    test_loader = data.load_test()

    logits,FVs,weights = carregar_dados(featurizer,train_loader)
    test_logits,test_features,weights = carregar_dados(featurizer,test_loader)


    passo1_costarr_original(logits, FVs, weights)
    scores_costarr, pred, modelo = passo2_costarr_corrigido(logits, FVs, weights)
    medias, logit_min, logit_max = passo3_funcs(logits, FVs, weights, pred)
    scores_funcs = passo4_funcs_test(medias,logit_min,logit_max,test_logits,test_features,weights,torch.max(test_logits,dim=1).indices)
    scores_costarr, _ = passo5_costarr_test(modelo,test_features,test_logits)
    passo6_comparar(scores_costarr, scores_funcs)
    resumo_final()