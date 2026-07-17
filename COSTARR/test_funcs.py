import torch
import pytest
from funcs import *

def testGNL():
    """Test GNL function"""
    ltmin = 0.2
    ltmax = 0.5
    logits = torch.tensor([[.4],[.2],[-0.1],[5]])
    print(GNL(ltmin,ltmax,logits))
    
def test_basic_mean_per_class():
    """Testa a média simples com 2 classes e 2 vetores cada."""
    concatVectors = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 0, 1, 1])

    result = calculateMeanConcatenatedVectors(concatVectors, targets)

    expected = torch.tensor([
        [2.0, 3.0],   # media da classe 0: ([1,2] + [3,4]) / 2
        [6.0, 7.0],   # media da classe 1: ([5,6] + [7,8]) / 2
    ], dtype=torch.float32)

    assert torch.allclose(result, expected), f"Esperado {expected}, obtido {result}"


def test_single_class():
    """Testa quando todos os vetores pertencem à mesma classe."""
    concatVectors = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 0])

    result = calculateMeanConcatenatedVectors(concatVectors, targets)

    expected = torch.tensor([
        [2.5, 3.5, 4.5],
    ], dtype=torch.float32)

    assert torch.allclose(result, expected), f"Esperado {expected}, obtido {result}"


def test_one_vector_per_class():
    """Testa quando cada classe tem apenas 1 vetor (a media e o proprio vetor)."""
    concatVectors = torch.tensor([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 1, 2])

    result = calculateMeanConcatenatedVectors(concatVectors, targets)

    expected = concatVectors.clone()  # media de 1 elemento = ele proprio

    assert torch.allclose(result, expected), f"Esperado {expected}, obtido {result}"


def test_three_classes_uneven_distribution():
    """Testa com 3 classes e distribuicao desigual de amostras."""
    concatVectors = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
        [6.0, 6.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 0, 0, 1, 1, 2])

    result = calculateMeanConcatenatedVectors(concatVectors, targets)

    # classe 0: ([1,1] + [2,2] + [3,3]) / 3 = [2, 2]
    # classe 1: ([4,4] + [5,5]) / 2 = [4.5, 4.5]
    # classe 2: ([6,6]) / 1 = [6, 6]
    expected = torch.tensor([
        [2.0, 2.0],
        [4.5, 4.5],
        [6.0, 6.0],
    ], dtype=torch.float32)

    assert torch.allclose(result, expected), f"Esperado {expected}, obtido {result}"


def test_non_consecutive_classes():
    """Testa com labels que nao sao consecutivas (ex: 0, 2, 5)."""
    concatVectors = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 0, 2, 2, 5, 5])

    result = calculateMeanConcatenatedVectors(concatVectors, targets)

    expected = torch.tensor([
        [1.5, 0.0],  # classe 0
        [3.5, 0.0],  # classe 2
        [5.5, 0.0],  # classe 5
    ], dtype=torch.float32)

    assert torch.allclose(result, expected), f"Esperado {expected}, obtido {result}"


def test_gradient_flows():
    """Verifica se os gradientes fluem corretamente (para retropropagacao)."""
    concatVectors = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 0, 1])

    result = calculateMeanConcatenatedVectors(concatVectors, targets)
    loss = result.sum()
    loss.backward()

    assert concatVectors.grad is not None, "Gradiente nao foi computado!"
    assert concatVectors.grad.shape == concatVectors.shape, (
        f"Shape do gradiente {concatVectors.grad.shape} != {concatVectors.shape}"
    )


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    aaa = torch.load("mnist_costarr.pt",weights_only=False)
    print(aaa["means"])
    print(calculateMagnitude(aaa["means"]))