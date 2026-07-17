from setuptools import setup, find_packages

setup(
    name="modulos_IC",  # Nome que aparecerá no pip list
    version="0.1",
    packages=find_packages(),      # Encontra automaticamente Utils, OutroModulo, etc.
    install_requires=[             # Opcional: dependências necessárias para seus módulos
        "torch",
        "torchvision",
        "numpy",
    ],
)