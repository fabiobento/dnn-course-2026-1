import os

import matplotlib as mpl
from torchvision import datasets


def apply_dlai_style():
    """
    Aplica um estilo global de plotagem e define um mapa de cores customizado.

    Returns:
        Uma tupla contendo o dicionário com o mapa de cores e o dicionário com o estilo de plotagem.
    """
    # Estilo global de plotagem
    PLOT_STYLE = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.family": "sans",  # "sans-serif",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }

    # mpl.rcParams.update(PLOT_STYLE)

    # Cores customizadas (reutilizáveis)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE


# Obtém o mapa de cores e as configurações de estilo
color_map, PLOT_STYLE = apply_dlai_style()
# Atualiza os parâmetros globais do matplotlib com o estilo definido
mpl.rcParams.update(PLOT_STYLE)


def get_dataset():
    """
    Garante a existência do diretório do dataset, faz o download do Fashion MNIST 
    (validação) se necessário e o retorna carregado.

    Returns:
        O objeto do dataset FashionMNIST do PyTorch.
    """
    # Define o caminho para o dataset
    path_dataset = "./dataset"

    # Se o caminho do dataset não existir, cria o diretório
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)

        # Baixa o dataset de validação do Fashion MNIST (via PyTorch)
        datasets.FashionMNIST(path_dataset, train=False, download=True)

    else:
        # Se o dataset já foi baixado, informa ao usuário
        print("O dataset já existe.")

    # Carrega o dataset de validação do Fashion MNIST (via PyTorch) sem baixá-lo novamente
    dataset = datasets.FashionMNIST(path_dataset, train=False, download=False)

    return dataset


def plot_counting(counting_params):
    """
    Gera um gráfico de barras mostrando a quantidade de parâmetros por camada da rede.

    Args:
        counting_params (dict): Um dicionário onde as chaves são os nomes das camadas 
                                e os valores são a quantidade de parâmetros.
    """
    import matplotlib.pyplot as plt

    # Plotagem do gráfico
    plt.figure(figsize=(14, 8))
    # Cria o gráfico de barras usando as chaves (nomes) e valores (contagem) do dicionário
    plt.bar(counting_params.keys(), counting_params.values())
    plt.xlabel("Nome da Camada")
    plt.ylabel("Número de Parâmetros")
    plt.title("Número de Parâmetros em Cada Camada Terminal")
    # Rotaciona os nomes das camadas no eixo X em 90 graus para evitar sobreposição
    plt.xticks(rotation=90)
    # Ajusta o layout para garantir que tudo caiba na tela sem cortes
    plt.tight_layout()
    plt.show()