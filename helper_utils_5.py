import os
import sys

import matplotlib.pyplot as plt
from directory_tree import DisplayTree
from fastai.vision.all import show_image, show_titled_image
from tqdm.auto import tqdm


def get_dataloader_bar(dataloader, color="green"):
    """
    Cria e configura uma barra de progresso para um dado dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): O dataloader que fornece o dataset.
        color (str): A cor da barra de progresso.

    Returns:
        pbar (tqdm.tqdm): O objeto da barra de progresso configurado.
    """
    # Obtém o número total de amostras a partir do dataset do dataloader.
    num_samples = len(dataloader.dataset)

    # Inicializa uma barra de progresso tqdm com as configurações especificadas.
    pbar = tqdm(
        # Define o número total de iterações para a barra.
        total=num_samples,
        # Calcula dinamicamente a largura da barra de progresso.
        ncols=int(num_samples / 10) + 300,
        # Define a string de formato para a aparência da barra de progresso.
        bar_format="{desc} {bar} {postfix}",
        # Direciona a saída da barra de progresso para o fluxo de saída padrão (stdout).
        file=sys.stdout,
        # Define a cor da barra de progresso.
        colour=color,
    )

    # Retorna o objeto da barra de progresso configurado.
    return pbar



def update_dataloader_bar(p_bar, batch, current_bs, n_samples):
    """
    Atualiza a barra de progresso existente com as informações do lote (batch) atual.

    Args:
        p_bar (tqdm.tqdm): O objeto da barra de progresso a ser atualizado.
        batch (int): O índice do lote atual.
        current_bs (int): O tamanho do lote atual (batch size).
        n_samples (int): O número total de amostras no dataset.
    """
    # Avança a barra de progresso pelo número de itens no lote atual.
    p_bar.update(current_bs)
    # Define a descrição para mostrar o número do lote atual.
    p_bar.set_description(f"Lote {batch+1}")

    # Verifica se o lote atual é o último.
    if (batch + 1) * current_bs > n_samples:
        # Atualiza o sufixo para mostrar o número total de amostras processadas.
        p_bar.set_postfix_str(f"{n_samples} de um total de  {n_samples} amostras")
    else:
        # Atualiza o sufixo para mostrar o número cumulativo de amostras processadas.
        p_bar.set_postfix_str(
            f"{current_bs*(batch+1)} de um total de  {n_samples} amostras"
        )



def plot_img(img, label=None, info=None, ax=None):
    """
    Plota uma imagem com rótulos opcionais e informações suplementares.

    Args:
        img (torch.Tensor ou numpy.ndarray): Os dados da imagem a serem plotados.
        label (str): Rótulo opcional para exibir como o título.
        info (str): Texto suplementar opcional para exibir abaixo da imagem.
        ax (matplotlib.axes.Axes): Eixos opcionais do matplotlib para plotar.
    """
    def add_info_text(ax, info):
        """
        Adiciona texto suplementar abaixo do gráfico em um determinado eixo.

        Args:
            ax (matplotlib.axes.Axes): O objeto de eixos do matplotlib.
            info (str): O texto a ser adicionado.
        """
        # Adiciona texto aos eixos em uma posição especificada.
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        # Define a posição do rótulo do eixo X para o topo.
        ax.xaxis.set_label_position("top")

    # Cria uma nova figura e eixos se nenhum for fornecido.
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Verifica se um rótulo foi fornecido para determinar como exibir la imagem.
    if label:
        # Cria uma string de título com o rótulo fornecido.
        title = f"Rótulo: {label}"
        # Exibe a imagem com o título gerado.
        show_titled_image((img, title), ax=ax)
    else:
        # Exibe a imagem sem um título.
        show_image(img, ax=ax)

    # Verifica se informações suplementares foram fornecidas.
    if info:
        # Adiciona a informação como texto abaixo da imagem.
        add_info_text(ax, info)

    # Se nenhum eixo foi passado, exibe o gráfico recém-criado.
    if ax is None:
        plt.show()



def get_grid(num_rows, num_cols, figsize=(16, 8)):
    """
    Cria uma grade de subplots e garante que o objeto de eixos seja formatado de forma consistente.

    Args:
        num_rows (int): O número de linhas na grade de subplots.
        num_cols (int): O número de colunas na grade de subplots.
        figsize (tuple): As dimensões da figura geral.

    Returns:
        fig (matplotlib.figure.Figure): O objeto da figura do matplotlib gerado.
        axes (list): A grade de eixos formatada, estruturada como um iterável ou lista 2D.
    """
    # Cria uma figura e um conjunto de subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Trata o caso onde há apenas uma linha.
    if num_rows == 1:
        # Garante que o objeto de eixos seja iterável para consistência.
        axes = [axes]
    # Trata o caso onde há apenas uma coluna.
    elif num_cols == 1:
        # Garante que o objeto de eixos seja uma lista 2D para indexação consistente.
        axes = [[ax] for ax in axes]
        
    # Retorna a figura e a grade de eixos formatada.
    return fig, axes



def print_data_folder_structure(root_dir, max_depth=1):
    """
    Imprime a estrutura de árvore de diretórios para um determinado diretório raiz.

    Args:
        root_dir (str): O caminho inicial para a árvore de diretórios.
        max_depth (int): A profundidade máxima para a varredura da árvore.
    """
    # Define as configurações para exibir a árvore de diretórios.
    config_tree = {
        # Especifica o caminho inicial para a árvore de diretórios.
        "dirPath": root_dir,
        # Define como False para incluir tanto arquivos quanto diretórios.
        "onlyDirs": False,
        # Define a profundidade máxima para a varredura da árvore.
        "maxDepth": max_depth,
        # Especifica uma opção de ordenação (100 normalmente significa sem ordenação específica).
        "sortBy": 100,
    }
    # Cria e exibe a estrutura da árvore usando a configuração descompactada.
    DisplayTree(**config_tree)


def explore_extensions(root_dir):
    """
    Explora uma árvore de diretórios para agrupar todos os caminhos de arquivos por suas extensões.

    Args:
        root_dir (str): O caminho do diretório inicial para a busca.

    Returns:
        extensions (dict): Um dicionário mapeando extensões de arquivos para uma lista de caminhos correspondentes.
    """
    # Inicializa um dicionário para armazenar os caminhos dos arquivos, agrupados por extensão.
    extensions = {}
    # Percorre a árvore de diretórios começando a partir do diretório raiz.
    for dirpath, _, filenames in os.walk(root_dir):
        # Itera sobre cada arquivo no diretório atual.
        for filename in filenames:
            # Extrai a extensão do arquivo e a converte para minúsculas.
            ext = os.path.splitext(filename)[1].lower()
            # Se a extensão ainda não foi vista antes, adiciona-a ao dicionário.
            if ext not in extensions:
                # Inicializa uma nova lista para esta extensão.
                extensions[ext] = []
            # Adiciona o caminho completo do arquivo à lista da sua respectiva extensão.
            extensions[ext].append(os.path.join(dirpath, filename))
    # Retorna o dicionário de extensões e seus caminhos de arquivo correspondentes.
    return extensions



def quick_debug(img):
    """
    Imprime informações básicas de depuração (debug) sobre um tensor de imagem.

    Args:
        img (torch.Tensor): O tensor de imagem a ser inspecionado.
    """
    # Imprime o formato (shape) do tensor de imagem.
    print(f"Formato: {img.shape}")  # Deve ser [3, 224, 224]
    # Imprime o tipo de dado do tensor.
    print(f"Tipo: {img.dtype}")  # Deve ser torch.float32
    # Imprime os valores mínimo e máximo de pixel no tensor.
    print(
        f"Intervalo de valores de pixel: [{img.min():.1f}, {img.max():.1f}]"
    )