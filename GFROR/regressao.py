import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Configurar o gerador para sempre dar o mesmo resultado (opcional)
np.random.seed(42)

# 2. Gerar dados aleatórios para X (100 pontos entre 0 e 10)
X = 300 * np.random.rand(100, 1)

# 3. Gerar dados para Y baseados na fórmula (Y = 2.5 * X + 10) + Ruído Aleatório
# O "np.random.randn" adiciona uma variação (ruído) para os pontos não ficarem perfeitamente alinhados
ruido = np.random.randn(100, 1) * 70
Y = 2.5 * X + 10 + ruido

# 4. Treinar o modelo de Regressão Linear para descobrir a linha ideal
modelo = LinearRegression()
modelo.fit(X, Y)
Y_pred = modelo.predict(X)

# 5. Gerar e exibir o gráfico
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', alpha=0.7, label='Dados de treinamento')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Linha de Regressão')

# Customização do gráfico
plt.title('Regressão Linear para predição de preço de imóveis')
plt.xlabel('Área')
plt.ylabel('Preço')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Mostra a imagem na tela
plt.savefig("regressao_linear_exemplo.pdf")