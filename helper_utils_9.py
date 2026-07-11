import itertools
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset


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
        
        # Update the dataset's class list to only include the target classes.
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


def visualise_images(loader, grid):
    """
    Visualiza uma grade de imagens aleatórias de um dataset, mostrando uma imagem por classe.

    Args:
        loader: O objeto DataLoader contendo o dataset.
        grid: Uma tupla especificando as dimensões da grade como (linhas, colunas).
    """
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
        
        # Redimensiona os valores dos pixels do intervalo normalizado para [0, 1] para visualização correta
        min_val = img_to_display.min()
        max_val = img_to_display.max()
        img_to_display = (img_to_display - min_val) / (max_val - min_val)
        
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
    
    
def verify_training_process(model_class, train_loader, loss_function, train_epoch_fn, device):
    """
    Verifica o processo de treinamento em um pequeno subconjunto de dados por algumas épocas.

    Args:
        model_class: A classe do modelo a ser instanciada para a verificação.
        train_loader: O DataLoader para o dataset de treino.
        loss_function: A função de perda a ser usada durante o treino.
        train_epoch_fn: A função que executa uma única época de treino.
        device: O dispositivo (ex: 'cuda' ou 'cpu') onde a verificação será executada.
    """
    # Imprime o cabeçalho do processo de verificação
    print("--- Verificando train_epoch (treinando por 5 épocas) ---\n")

    # Define o número de épocas e lotes (batches) para a execução de verificação
    NUM_VERIFY_EPOCHS = 5
    NUM_VERIFY_BATCHES = 10

    # Instancia o modelo e o move para o dispositivo especificado
    verify_model = model_class(15).to(device)
    # Inicializa o otimizador Adam com uma taxa de aprendizado específica
    verify_optimizer = optim.Adam(verify_model.parameters(), lr=0.0005)

    # Cria um pequeno subconjunto de dados de treino para verificação rápida
    batches = list(itertools.islice(iter(train_loader), NUM_VERIFY_BATCHES))
    # Concatena as imagens e rótulos dos lotes selecionados
    all_images = torch.cat([b[0] for b in batches])
    all_labels = torch.cat([b[1] for b in batches])
    # Cria um TensorDataset e um DataLoader para esse subconjunto
    verify_subset_dataset = TensorDataset(all_images, all_labels)
    verify_subset_loader = DataLoader(verify_subset_dataset, batch_size=train_loader.batch_size)

    # Clona os pesos iniciais de uma camada específica para verificar alterações posteriores
    initial_weight = verify_model.conv_block1.block[0].weight.clone()
    # Inicializa uma lista para armazenar a perda de cada época
    epoch_losses = []

    print(f"Treinando em {len(verify_subset_dataset)} imagens por {NUM_VERIFY_EPOCHS} épocas:\n")
    # Loop pelas épocas definidas para a verificação
    for epoch in range(NUM_VERIFY_EPOCHS):
        # Executa uma única época de treino e obtém a perda
        loss = train_epoch_fn(
            model=verify_model,
            train_loader=verify_subset_loader,
            loss_function=loss_function,
            optimizer=verify_optimizer,
            device=device
        )
        # Adiciona a perda à lista e imprime o resultado da época
        epoch_losses.append(loss)
        print(f"Época [{epoch+1}/{NUM_VERIFY_EPOCHS}], Perda: {loss:.4f}")

    # Obtém os pesos da mesma camada após o término do treinamento
    trained_weight = verify_model.conv_block1.block[0].weight

    # Verifica se os pesos mudaram em relação aos seus valores iniciais
    weights_changed = not torch.equal(initial_weight, trained_weight)
    if weights_changed:
        print("\nChecagem de Atualização de Pesos:\tOs pesos do modelo mudaram durante o treino.")
    else:
        print("\nChecagem de Atualização de Pesos:\tOs pesos do modelo NÃO mudaram.")

    # Verifica se a perda final é menor do que a perda inicial
    loss_decreased = epoch_losses[-1] < epoch_losses[0]
    if loss_decreased:
        print(f"Checagem de Tendência da Perda:\tA perda diminuiu de {epoch_losses[0]:.4f} para {epoch_losses[-1]:.4f}.")
    else:
        print(f"Checagem de Tendência da Perda:\tA perda NÃO apresentou tendência de queda.")
        
        
def verify_validation_process(model_class, val_loader, loss_function, validate_epoch_fn, device):
    """
    Verifica o processo de validação em um pequeno subconjunto de dados.

    Args:
        model_class: A classe do modelo a ser instanciada para a verificação.
        val_loader: O DataLoader para o dataset de validação.
        loss_function: A função de perda a ser usada durante a validação.
        validate_epoch_fn: A função que executa uma única época de validação.
        device: O dispositivo (ex: 'cuda' ou 'cpu') onde a verificação será executada.
    """
    # Imprime o cabeçalho do processo de verificação
    print("--- Verificando validate_epoch ---\n")

    # Define o número de lotes para a execução de verificação
    NUM_VERIFY_BATCHES = 10

    # Instancia o modelo e o move para o dispositivo especificado
    verify_model = model_class(15).to(device)

    # Cria um pequeno subconjunto de dados de validação para verificação rápida
    val_batches = list(itertools.islice(iter(val_loader), NUM_VERIFY_BATCHES))
    # Concatena as imagens e rótulos dos lotes selecionados
    val_all_images = torch.cat([b[0] for b in val_batches])
    val_all_labels = torch.cat([b[1] for b in val_batches])
    # Cria um TensorDataset e um DataLoader para esse subconjunto
    verify_val_subset_dataset = TensorDataset(val_all_images, val_all_labels)
    verify_val_subset_loader = DataLoader(verify_val_subset_dataset, batch_size=val_loader.batch_size)

    # Clona os pesos iniciais de uma camada específica para verificar se ocorrem alterações indevidas
    initial_weight = verify_model.conv_block1.block[0].weight.clone()

    print(f"Validando em {len(verify_val_subset_dataset)} imagens:\n")
    # Executa uma única época de validação no subconjunto e obtém os retornos
    val_loss, val_accuracy = validate_epoch_fn(
        model=verify_model,
        val_loader=verify_val_subset_loader,
        loss_function=loss_function,
        device=device
    )

    # Obtém os pesos da mesma camada após a execução da função de validação
    validated_weight = verify_model.conv_block1.block[0].weight

    # Imprime a perda e acurácia retornadas
    print(f"Perda de Validação Retornada: {val_loss:.4f}")
    print(f"Acurácia de Validação Retornada: {val_accuracy:.2f}%\n")

    # Verifica se a perda e a acurácia retornadas são do tipo de dado correto (float)
    types_correct = isinstance(val_loss, float) and isinstance(val_accuracy, float)
    if types_correct:
        print("\nChecagem dos Tipos de Retorno:\tA função retornou floats para perda e acurácia.")
    else:
        print("\nChecagem dos Tipos de Retorno:\tA função NÃO retornou os tipos de dados corretos.")

    # Verifica se os pesos permaneceram inalterados durante a validação
    weights_unchanged = torch.equal(initial_weight, validated_weight)
    if weights_unchanged:
        print("Checagem de Integridade dos Pesos:\tOs pesos do modelo não foram alterados durante a validação.")
    else:
        print("Checagem de Integridade dos Pesos:\tOs pesos do modelo FORAM ALTERADOS.")