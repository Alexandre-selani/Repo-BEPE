from scipy.stats import wilcoxon

openmax = [0.930,0.885,0.940,0.953,0.877]
opengan = [0.629,0.861,0.840,0.718,0.205]

print(wilcoxon(openmax,opengan))
