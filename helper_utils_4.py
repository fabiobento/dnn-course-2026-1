import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

# Referência de texto para validação da decodificação
letter_ref = [
    "Dear Laurence",
    "Hope the PyTorch course is going well",
    "Do notforget to keep the labs interesting and engaging",
    "Maybe the students could decode my messy handwriting",
    "That might be a bit too challenging though",
    "I am impressed you are able to read this",
]

path_data = "./EMNIST_data"

def load_hidden_message_images(file_name="hidden_message_images.pkl"):
    """
    Carrega imagens de mensagens ocultas de um arquivo pickle (.pkl).

    Args:
        file_name (str): O nome do arquivo para carregar as imagens.

    Returns:
        message_imgs (list): Uma lista contendo as imagens carregadas.
    """
    # Abre o arquivo especificado no modo de leitura binária
    with open(file_name, "rb") as f:
        import pickle
        # Carrega os dados serializados do arquivo
        message_imgs = pickle.load(f)
        
    return message_imgs

def decode_word_imgs(word_imgs, model, device):
    """
    Decodifica uma sequência de imagens de caracteres em uma única string de palavra 
    usando um modelo de classificação fornecido.

    Args:
        word_imgs (list): Uma coleção de tensores de imagem representando 
            caracteres individuais de uma palavra.
        model (torch.nn.Module): O modelo de rede neural treinado.
        device (torch.device): O dispositivo de computação (CPU/GPU).

    Returns:
        decoded_word (str): A string concatenada dos caracteres previstos.
    """
    # Define o modelo para modo de avaliação (desativa dropout, etc.)
    model.eval()
    
    # Lista para armazenar os caracteres previstos
    decoded_chars = []
    
    # Desativa o cálculo de gradientes para inferência (economiza memória)
    with torch.no_grad():
        for char_img in word_imgs:
            # Adiciona a dimensão do lote (batch) e move para o dispositivo alvo
            # Transforma [1, 28, 28] em [1, 1, 28, 28]
            char_img = char_img.unsqueeze(0).to(device) 
            
            # Passagem para frente (forward pass) para prever as probabilidades
            output = model(char_img)
            
            # Extrai o índice da classe com a maior probabilidade
            _, predicted = output.max(1)
            
            # Recupera o valor numérico da previsão
            predicted_label = predicted.item()
            
            # Converte o rótulo previsto para a letra minúscula correspondente
            # No EMNIST Letters, 0 costuma mapear para 'a'
            lowercase_char = chr(ord("a") + predicted_label)
            
            decoded_chars.append(f"{lowercase_char}")
            
    # Junta a lista de caracteres individuais para formar a palavra final
    decoded_word = "".join(decoded_chars)
    
    return decoded_word

def visualize_image(img, label=None, ax=None):
    """
    Visualiza uma imagem do EMNIST com seu rótulo. Se um eixo (ax) for fornecido, 
    plota nele; caso contrário, cria uma nova figura.
    """
    # Verifica se a imagem é um tensor e converte para array numpy
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze()
    # Trata arrays numpy de 3 dimensões (ex: canais de cor)
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img[:, :, 0]

    # Se um rótulo for fornecido, converte para caracteres legíveis
    if label is not None:
        uppercase_char, lowercase_char = convert_emnist_label_to_char(label)
        title = f"Letra EMNIST: {uppercase_char}/{lowercase_char}"
    else:
        title = None

    # Gerencia a criação da figura ou uso de eixo existente
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        show_colorbar = True
    else:
        show_colorbar = False

    # Exibe a imagem em escala de cinza
    im = ax.imshow(img, cmap="gray")
    
    # Configurações de grade e marcações (ticks) de 0 a 27 pixels
    ax.set_xticks(np.arange(0, 28, 1))
    ax.set_yticks(np.arange(0, 28, 1))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="red", alpha=0.3)
    
    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax)
        plt.show()

def display_data_loader_contents(data_loader):
    """
    Exibe informações sobre o conteúdo do DataLoader (tamanhos e formatos).
    """
    try:
        print("Total de imagens no dataset:", len(data_loader.dataset))
        print("Total de lotes (batches):", len(data_loader))
        
        # Analisa apenas o primeiro lote para exemplo
        for batch_idx, (data, labels) in enumerate(data_loader):
            print(f"--- Lote {batch_idx + 1} ---")
            print(f"Formato dos dados (Data shape): {data.shape}")
            print(f"Formato dos rótulos (Labels shape): {labels.shape}")
            break
            
    except StopIteration:
        print("O DataLoader está vazio.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

def evaluate_per_class(model, test_loader, device):
    """
    Avalia a acurácia do modelo para cada classe individual (letra).
    """
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Ajuste de índice se necessário (dependendo do mapeamento 0-25 ou 1-26)
            targets = targets - 1

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    class_accuracies = {}

    # Calcula acurácia para cada uma das 26 letras do alfabeto
    for class_idx in range(26):
        # Filtra previsões e alvos específicos desta classe
        class_targets = [t for t, p in zip(all_targets, all_predictions) if t == class_idx]
        class_predictions = [p for t, p in zip(all_targets, all_predictions) if t == class_idx]

        if len(class_targets) > 0:
            class_accuracies[chr(65 + class_idx)] = accuracy_score(class_targets, class_predictions)
        else:
            class_accuracies[chr(65 + class_idx)] = 0.0

    return class_accuracies

def save_student_model(model, filename="trained_student_model.pth"):
    """
    Salva o estado do modelo treinado em um arquivo.
    """
    save_dict = {"model": model}
    torch.save(save_dict, filename)
    print(f"Modelo salvo com sucesso em: {filename}")

def convert_emnist_label_to_char(label):
    """
    Converte um rótulo numérico do EMNIST para caracteres maiúsculos e minúsculos.
    """
    # O EMNIST Letters usa 1-26 para A-Z
    if not (1 <= label <= 26):
        raise ValueError("O rótulo deve estar entre 1 e 26 inclusive.")

    # 65 é o código ASCII para 'A', 97 para 'a'
    uppercase_char = chr(64 + label)
    lowercase_char = chr(96 + label)

    return (uppercase_char, lowercase_char)