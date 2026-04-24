import random
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

def display_image(image, label, title, num_ticks=6, show_values=True):
    """
    Exibe a imagem com seu respectivo rótulo e título.

    Esta função lida com diferentes formatos de imagem (PIL Image e PyTorch Tensor),
    normaliza o intervalo de exibição e, opcionalmente, sobrepõe os valores numéricos 
    dos pixels na imagem.

    Args:
        image: Os dados da imagem a serem exibidos. Pode ser uma imagem PIL ou um Tensor PyTorch.
        label: O rótulo (classe) associado à imagem.
        title: O título para o gráfico.
        num_ticks (int, opcional): O número de marcações na barra de cores. Padrão é 6.
        show_values (bool, opcional): Se True, sobrepõe o valor numérico de cada pixel na imagem. Padrão é True.
    """
    # Inicializa variáveis para o intervalo de valores e dados da imagem.
    vmin_val, vmax_val = None, None
    image_data = None

    # Verifica se a entrada é uma imagem PIL.
    if isinstance(image, Image.Image):
        # Define o intervalo de valores para uma imagem padrão de 8 bits.
        vmin_val = 0
        vmax_val = 255
        # Converte a imagem PIL para um array NumPy.
        image_data = np.array(image)
    # Verifica se a entrada é um Tensor PyTorch.
    elif isinstance(image, torch.Tensor):
        # Converte o tensor para um array NumPy e remove dimensões extras de tamanho 1.
        image_np = image.numpy().squeeze()
        # Determina os valores mín e máx do tensor para normalização da escala de cores.
        vmin_val = image_np.min()
        vmax_val = image_np.max()
        # Atribui o array NumPy para image_data.
        image_data = image_np
    # Trata tipos de imagem não suportados.
    else:
        print("Aviso: Tipo de imagem não suportado.")
        return

    # Cria uma nova figura para o gráfico.
    plt.figure(figsize=(9, 9))
    # Exibe os dados da imagem em escala de cinza.
    plt.imshow(image_data, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    # Define o título do gráfico com o título fornecido e o rótulo.
    plt.title(f'{title} | Rótulo (Label): {label}')

    # Verifica se os valores dos pixels devem ser exibidos sobre a imagem.
    if show_values:
        # Calcula um limite para determinar a cor do texto (preto ou branco) para contraste.
        threshold = (vmin_val + vmax_val) / 2.0
        # Obtém as dimensões da imagem.
        height, width = image_data.shape
        
        # Itera sobre cada pixel para exibir seu valor.
        for y in range(height):
            for x in range(width):
                # Obtém o valor do pixel.
                value = image_data[y, x]
                # Define a cor do texto com base no brilho do pixel.
                text_color = "white" if value < threshold else "black"
                
                # Formata o texto para exibição, tratando inteiros e decimais de forma diferente.
                text_to_display = f"{value:.0f}" if isinstance(value, np.integer) else f"{value:.1f}"
                
                # Adiciona o valor do pixel como texto ao gráfico.
                plt.text(x, y, text_to_display, 
                         ha="center", va="center", color=text_color, fontsize=6)

    # Adiciona uma grade ao gráfico para facilitar a contagem de pixels.
    plt.grid(True, color='red', alpha=0.3, zorder=2)
    # Define as marcações do eixo x.
    plt.xticks(np.arange(0, 28, 4))
    # Define as marcações do eixo y.
    plt.yticks(np.arange(0, 28, 4))
    
    # Adiciona uma barra de cores lateral.
    cbar = plt.colorbar()
    # Cria marcações uniformemente espaçadas para a barra de cores.
    ticks = np.linspace(vmin_val, vmax_val, num=num_ticks)
    # Define as marcações na barra de cores.
    cbar.set_ticks(ticks)
    # Formata os rótulos das marcações na barra de cores.
    cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])

    # Mostra o gráfico final.
    plt.show()

def display_predictions(model, test_loader, device):
    """
    Exibe uma grade de previsões para uma amostra aleatória de cada classe.

    Args:
        model: O modelo PyTorch treinado.
        test_loader: O DataLoader para o conjunto de teste.
        device: O dispositivo (ex: 'cuda' ou 'cpu') para executar a inferência.
    """
    # Garante que o modelo esteja no dispositivo especificado e em modo de avaliação.
    model.to(device)
    model.eval()

    # Cria um dicionário para armazenar os índices de cada classe (0-9).
    class_indices = {i: [] for i in range(10)}
    
    # Preenche o dicionário com os índices de todas as amostras para cada classe.
    for idx, (_, label) in enumerate(test_loader.dataset):
        class_indices[label].append(idx)
        
    # Seleciona um índice aleatório da lista de índices de cada classe.
    random_indices = [random.choice(indices) for indices in class_indices.values()]
    
    # Recupera as imagens e rótulos correspondentes usando os índices selecionados.
    sample_images = torch.stack([test_loader.dataset[i][0] for i in random_indices])
    sample_labels = [test_loader.dataset[i][1] for i in random_indices]

    # Desativa temporariamente o cálculo de gradientes para a inferência.
    with torch.no_grad():
        # Passa as imagens selecionadas pelo modelo para obter as saídas.
        outputs = model(sample_images.to(device))
        # Obtém a classe prevista para cada imagem (índice do maior valor).
        _, predictions = torch.max(outputs, 1)

    # Cria uma figura e uma grade de subplots (2 linhas, 5 colunas).
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    # Define um título principal para toda a figura.
    fig.suptitle('Previsões do Modelo para uma Amostra de Cada Classe', fontsize=16)

    # Itera pelos subplots para exibir cada imagem e sua previsão.
    for i, ax in enumerate(axes.flat):
        # Extrai a imagem, o rótulo real e o rótulo previsto.
        image = sample_images[i].cpu().squeeze()
        true_label = sample_labels[i]
        predicted_label = predictions[i].item()

        # Exibe a imagem no subplot atual.
        ax.imshow(image, cmap='gray')
        
        # Define a cor do título: verde se correto, vermelho se incorreto.
        title_color = 'green' if true_label == predicted_label else 'red'
        ax.set_title(f"Real: {true_label}\nPred: {predicted_label}", color=title_color)
        
        # Oculta os eixos para uma visualização mais limpa.
        ax.axis('off')

    # Ajusta o layout para evitar sobreposição.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Ajusta o espaçamento vertical entre os subplots.
    plt.subplots_adjust(hspace=0.3)
    # Exibe o gráfico.
    plt.show()

def plot_metrics(train_loss, test_acc):
    """
    Exibe gráficos lado a lado da perda de treinamento e acurácia de teste por épocas.

    Args:
        train_loss (list): Lista com a perda média de treinamento para cada época.
        test_acc (list): Lista com a acurácia de teste para cada época.
    """
    # Define o número de épocas com base no tamanho da lista de perda.
    num_epochs = len(train_loss)
    # Cria um intervalo de épocas começando em 1 para o eixo x.
    epochs = range(1, num_epochs + 1)

    # Cria uma figura com dois subplots (1 linha, 2 colunas).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Gráfico 1: Perda de Treinamento (Training Loss) ---
    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('Perda de Treinamento por Época', fontsize=14)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Perda (Loss)', fontsize=12)
    ax1.grid(True)
    # Garante que as marcações do eixo x sejam números inteiros.
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Gráfico 2: Acurácia de Teste (Test Accuracy) ---
    ax2.plot(epochs, test_acc, marker='o', linestyle='-', color='red')
    ax2.set_title('Acurácia de Teste por Época', fontsize=14)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Acurácia (%)', fontsize=12)
    ax2.grid(True)
    # Garante que as marcações do eixo x sejam números inteiros.
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Ajusta o layout e exibe os gráficos.
    plt.tight_layout()
    plt.show()