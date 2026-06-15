import copy
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    """
    Carrega e filtra o dataset CIFAR-100 para incluir apenas as classes-alvo especificadas.

    Esta função primeiro verifica se existe uma cópia local do dataset CIFAR-100 e
    realiza o download caso não seja encontrado. Em seguida, filtra os conjuntos de
    treino e teste para reter apenas as imagens e rótulos correspondentes às classes
    especificadas em `target_classes`. Os rótulos são remapeados para serem contíguos a partir de 0.

    Args:
        target_classes: Uma lista de strings com os nomes das classes a serem incluídas no subconjunto.
        train_transform: Uma transformação do torchvision para ser aplicada às imagens de treino.
        val_transform: Uma transformação do torchvision para ser aplicada às imagens de teste/validação.
        root: O diretório raiz onde o dataset está armazenado ou será baixado.

    Returns:
        Uma tupla contendo o dataset de treino filtrado e o dataset de teste filtrado.
        Retorna (None, None) se uma classe-alvo especificada não for encontrada.
    """
    # Constrói o caminho para o diretório do dataset CIFAR-100.
    cifar100_path = os.path.join(root, 'cifar-100-python')
    # Verifica se o diretório do dataset existe localmente.
    if os.path.isdir(cifar100_path):
        print(f"Dataset encontrado em '{root}'. Carregando dos arquivos locais.")
    # Se não for encontrado, informa ao usuário que o download será realizado.
    else:
        print(f"Dataset não encontrado em '{root}'. Baixando...")

    # Carrega o dataset completo de treino do CIFAR-100.
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Carrega o dataset completo de teste do CIFAR-100.
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=False, 
        download=True, 
        transform=val_transform
    )
    print("Dataset carregado com sucesso.")

    # Obtém a lista de todos os nomes das classes do dataset.
    all_classes = train_dataset_full.classes
    try:
        # Obtém os índices inteiros originais para os nomes das classes-alvo.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Trata o caso onde um nome de classe especificado não existe no dataset.
    except ValueError as e:
        print(f"Erro: Uma das classes-alvo não foi encontrada no CIFAR-100. {e}")
        return None, None
        
    # Cria um mapeamento dos índices originais das classes para novos índices contíguos (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    # Define uma função auxiliar para filtrar um dataset com base nas classes-alvo.
    def _filter_dataset(dataset):
        # Converte a lista de alvos em um array NumPy para indexação booleana eficiente.
        targets_np = np.array(dataset.targets)
        # Cria uma máscara booleana para identificar quais amostras pertencem às classes-alvo.
        indices_to_keep = np.isin(targets_np, target_indices)
        
        # Filtra os dados de imagem do dataset usando a máscara booleana.
        dataset.data = dataset.data[indices_to_keep]
        
        # Obtém os rótulos originais das amostras que serão mantidas.
        original_targets_to_keep = targets_np[indices_to_keep]
        # Remapeia os rótulos originais para os novos rótulos contíguos.
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        
        # Atualiza a lista de classes do dataset para incluir apenas as classes-alvo.
        dataset.classes = target_classes
        return dataset

    print(f"\nFiltrando para {len(target_classes)} classes...")
    # Aplica a lógica de filtragem ao dataset de treino completo.
    train_dataset_subset = _filter_dataset(train_dataset_full)
    # Aplica a lógica de filtragem ao dataset de teste completo.
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtragem concluída. Retornando os datasets de treino e validação.")
    
    # Retorna os subconjuntos filtrados de treino e teste.
    return train_dataset_subset, test_dataset_subset


def visualise_images(dataset, grid):
    """
    Exibe uma grade de imagens de um dataset, com uma imagem aleatória por classe.

    Args:
        dataset: O objeto do dataset contendo as imagens e rótulos.
        grid (tuple): Uma tupla especificando o número de linhas e colunas para a grade de imagens.
    """

    # Cria uma cópia rasa (shallow copy) do dataset para evitar modificar o original
    dataset_copy = copy.copy(dataset)
    # Define a transformação na cópia do dataset para converter as imagens em tensores
    dataset_copy.transform = torchvision.transforms.ToTensor()

    # Cria um DataLoader para gerenciar o agrupamento em lotes (batching) e o embaralhamento dos dados
    loader = DataLoader(dataset_copy, batch_size=64, shuffle=True)

    # Desempacota as dimensões da grade a partir da tupla de entrada
    rows, cols = grid
    # Calcula o número total de imagens a serem exibidas na grade
    num_images_to_show = rows * cols

    # Obtém o objeto do dataset a partir do DataLoader
    dataset_to_show = loader.dataset

    # Cria um dicionário para armazenar listas de índices para cada classe
    class_indices = defaultdict(list)
    # Itera pelo dataset para preencher o dicionário class_indices
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Obtém a lista de nomes das classes do dataset
    class_names = dataset_to_show.classes

    # Cria uma figura e um conjunto de subplots para o layout da grade
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Itera sobre cada subplot na grade
    for i, ax in enumerate(axes.flat):
        # Se o índice atual estiver fora dos limites, desativa o eixo do subplot
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Define o rótulo da classe com base no índice da iteração atual
        class_label = i
        
        # Obtém a lista de índices de imagem para a classe atual
        indices_for_class = class_indices[class_label]
        # Se não houver imagens para esta classe, desativa o eixo do subplot
        if not indices_for_class:
            ax.axis('off')
            continue

        # Escolhe um índice de imagem aleatório da lista para a classe atual
        random_image_index = random.choice(indices_for_class)
        
        # Recupera o tensor da imagem e seu rótulo correspondente do dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Converte o tensor em um array NumPy e transpõe as dimensões para exibição (H, W, C)
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Obtém o nome da classe correspondente ao rótulo da classe
        class_name = class_names[class_label]
        
        # Exibe a imagem no subplot atual
        ax.imshow(img_to_display)
        
        # Define o título do subplot com o nome da classe capitalizado
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Desativa os eixos para uma aparência mais limpa
        ax.axis('off')

    # Ajusta os parâmetros dos subplots para um layout compacto
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Exibe o gráfico
    plt.show()

    # Limpa a cópia do dataset para liberar memória
    del dataset_copy
    
    
def plot_training_metrics(metrics):
    """
    Plota as métricas de treino e validação do processo de treinamento de um modelo.

    Esta função gera dois gráficos lado a lado:
    1. Perda de Treino vs. Perda de Validação.
    2. Acurácia de Validação.

    Args:
        metrics (list): Uma lista ou tupla contendo três listas:
                        [train_losses, val_losses, val_accuracies].
    """
    # Desempacota as métricas em suas respectivas listas
    train_losses, val_losses, val_accuracies = metrics
    
    # Determina o número de épocas a partir do tamanho da lista de perdas de treino
    num_epochs = len(train_losses)
    # Cria um intervalo indexado em 1 com os números das épocas para o eixo X
    epochs = range(1, num_epochs + 1)

    # Cria uma figura e um conjunto de subplots com 1 linha e 2 colunas
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configura o primeiro subplot para a perda de treino e validação ---
    # Seleciona o primeiro subplot
    ax1 = axes[0]
    # Plota os dados da perda de treino
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Perda de Treino')
    # Plota os dados da perda de validação
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Perda de Validação')
    # Define o título e os rótulos dos eixos para o gráfico de perda
    ax1.set_title('Perda de Treino e Validação', fontsize=14)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Perda', fontsize=12)
    # Exibe a legenda
    ax1.legend()
    # Adiciona uma grade para melhor legibilidade
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configura o segundo subplot para a acurácia de validação ---
    # Seleciona o segundo subplot
    ax2 = axes[1]
    # Plota os dados da acurácia de validação
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Acurácia de Validação')
    # Define o título e os rótulos dos eixos para o gráfico de acurácia
    ax2.set_title('Acurácia de Validação', fontsize=14)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Acurácia (%)', fontsize=12)
    # Exibe a legenda
    ax2.legend()
    # Adiciona uma grade para melhor legibilidade
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Aplica estilização dinâmica e consistente a ambos os subplots ---
    # Calcula um intervalo adequado para as marcações do eixo X para evitar poluição visual
    x_interval = (num_epochs - 1) // 10 + 1

    # Percorre cada subplot para aplicar as configurações comuns de eixo
    for ax in axes:
        # Define o eixo Y para iniciar em 0 e o eixo X para abranger as épocas
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Define o localizador de marcações principais para o eixo X usando o intervalo dinâmico
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Define o tamanho da fonte para os rótulos das marcações em ambos os eixos
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Ajusta os parâmetros dos subplots para um layout compacto
    plt.tight_layout()
    # Exibe os gráficos
    plt.show()
    
    
def visualise_predictions(model, data_loader, device, grid):
    """
    Visualiza as previsões do modelo em uma grade de imagens de um dataset.

    Args:
        model: O modelo PyTorch treinado a ser usado para as previsões.
        data_loader: O DataLoader do PyTorch para o dataset.
        device: O dispositivo (ex: 'cpu' ou 'cuda') onde o modelo será executado.
        grid (tuple): Uma tupla especificando o número de linhas e colunas para a grade de imagens.
    """
    # Define o modelo para modo de avaliação
    model.eval()

    # Obtém o dataset e os nomes das classes a partir do carregador de dados
    dataset = data_loader.dataset
    class_names = dataset.classes
    
    # Define os valores de média e desvio padrão para desnormalizar as imagens
    cifar100_mean = np.array([0.5071, 0.4867, 0.4408])
    cifar100_std = np.array([0.2675, 0.2565, 0.2761])
    
    # Cria um dicionário para armazenar listas de índices para cada classe
    class_indices = defaultdict(list)
    # Itera pelo dataset para preencher o dicionário class_indices
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)
        
    # Desempacota as dimensões da grade
    rows, cols = grid
    # Calcula o número total de imagens a serem exibidas
    num_images_to_show = rows * cols
    
    # Cria uma figura e um conjunto de subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2)) 
    # Ajusta o espaçamento entre os subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.8)

    # Itera sobre cada subplot na grade
    for i, ax in enumerate(axes.flat):
        # Se o índice atual estiver fora dos limites, desativa o eixo do subplot
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Define o rótulo da classe com base no índice da iteração atual
        class_label = i
        
        # Obtém a lista de índices de imagem para la classe atual
        indices_for_class = class_indices[class_label]
        # Se não houver imagens para esta classe, desativa o eixo do subplot
        if not indices_for_class:
            ax.axis('off')
            continue

        # Escolhe um índice de imagem aleatório da lista para a classe atual
        random_image_index = random.choice(indices_for_class)
        # Recupera o tensor da imagem e seu rótulo real
        image_tensor, true_label = dataset[random_image_index]
        
        # Adiciona uma dimensão de lote (batch) e move o tensor para o dispositivo especificado
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Desativa o cálculo de gradientes para a inferência
        with torch.no_grad():
            # Obtém as previsões do modelo
            output = model(image_batch)
            # Encontra o índice da maior pontuação, que representa a classe prevista
            _, predicted_index = torch.max(output, 1)
        
        # Extrai o rótulo previsto como um número nativo do Python
        predicted_label = predicted_index.item()
        
        # Converte o tensor em um array NumPy e transpõe as dimensões para exibição
        img_np = image_tensor.cpu().numpy().transpose((1, 2, 0))
        # Desnormaliza a imagem usando a média e desvio padrão predefinidos
        denormalized_img = cifar100_std * img_np + cifar100_mean
        # Limita os valores dos pixels para o intervalo válido [0, 1]
        clipped_img = np.clip(denormalized_img, 0, 1)
        
        # Obtém os nomes em string para os rótulos real e previsto
        true_name = class_names[true_label]
        predicted_name = class_names[predicted_label]
        
        # Define a cor do título como verde para previsões corretas e vermelha para incorretas
        title_color = 'green' if true_label == predicted_label else 'red'
        
        # Exibe a imagem
        ax.imshow(clipped_img)
        # Define o título exibindo os rótulos Real e Previsto
        ax.set_title(f"Real: {true_name.capitalize()}\nPrev: {predicted_name.capitalize()}", 
                     color=title_color, fontsize=10, pad=5)
        # Desativa o eixo
        ax.axis('off')

    # Ajusta os parâmetros dos subplots para um layout compacto
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Exibe o gráfico final
    plt.show()