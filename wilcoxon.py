import itertools
import pandas as pd
from scipy.stats import spearmanr

# linhas = métodos, colunas = benchmarks
f1 = pd.DataFrame(
    {
        "MNIST/Omniglot": [0.890, 0.912, 0.839, 0.908, 0.719, 0.909],
        "TinyImageNet":   [0.611, 0.640, 0.650, 0.617, 0.524, 0.484],
        "Panicum":        [0.936, 0.917, 0.937, 0.882, 0.650, 0.787],
        "Ceratocystis":   [0.934, 0.937, 0.947, 0.946, 0.577, 0.686],
    },
    index=["MSP", "OpenMax", "CAC", "COSTARR", "OpenGAN", "GFROR"],
)

rows = []
for a, b in itertools.combinations(f1.columns, 2):
    rho, p = spearmanr(f1[a], f1[b])
    rows.append({"pair": f"{a} vs. {b}", "rho": round(rho, 2), "p": round(p, 3)})

print(pd.DataFrame(rows).to_string(index=False))
