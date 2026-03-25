import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def plot_data(distances, times, normalize=False):
    """
    Cria um gráfico de dispersão dos pontos de dados.

    Args:
        distances: Os pontos de dados de entrada para o eixo x (distâncias).
        times: Os pontos de dados de destino para o eixo y (tempos).
        normalize: Um sinalizador booleano indicando se os dados estão normalizados.
    """
    # Cria uma nova figura com um tamanho especificado
    plt.figure(figsize=(8, 6))

    # Plota os pontos de dados como um gráfico de dispersão
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Tempos de Entrega Reais')

    # Verifica se os dados estão normalizados para definir rótulos e título apropriados
    if normalize:
        # Define o título do gráfico para dados normalizados
        plt.title('Dados de Entrega Normalizados (Bikes & Carros)')
        # Define o rótulo do eixo x para dados normalizados
        plt.xlabel('Distância Normalizada')
        # Define o rótulo do eixo y para dados normalizados
        plt.ylabel('Tempo Normalizado')
        # Exibe a legenda
        plt.legend()
        # Adiciona uma grade ao gráfico
        plt.grid(True)
        # Mostra o gráfico
        plt.show()

    # Trata o caso de dados não normalizados
    else:
        # Define o título do gráfico para dados originais
        plt.title('Dados de Entrega (Bikes & Carros)')
        # Define o rótulo do eixo x para dados originais
        plt.xlabel('Distância (milhas)')
        # Define o rótulo do eixo y para dados originais
        plt.ylabel('Tempo (minutos)')
        # Exibe a legenda
        plt.legend()
        # Adiciona uma grade ao gráfico
        plt.grid(True)
        # Mostra o gráfico
        plt.show()

def plot_final_fit(model, distances, times, distances_norm, times_std, times_mean):
    """
    Plota as previsões de um modelo treinado contra os dados originais,
    após desnormalizar as previsões.

    Args:
        model: O modelo treinado usado para a previsão.
        distances: Os dados de entrada originais, não normalizados.
        times: Os dados de destino originais, não normalizados.
        distances_norm: Os dados de entrada normalizados para o modelo.
        times_std: O desvio padrão usado para desnormalização.
        times_mean: O valor médio usado para desnormalização.
    """
    # Define o modelo para modo de avaliação
    model.eval()

    # Desativa o cálculo de gradientes para a previsão
    with torch.no_grad():
        # Obtém as previsões do modelo usando dados normalizados
        predicted_norm = model(distances_norm)

    # Desnormaliza as previsões para sua escala original
    predicted_times = (predicted_norm * times_std) + times_mean

    # Cria uma nova figura para o gráfico
    plt.figure(figsize=(8, 6))

    # Plota os pontos de dados originais
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Dados Reais (Bikes & Carros)')

    # Plota as previsões desnormalizadas do modelo
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', label='Previsões do Modelo Não Linear')

    # Define o título do gráfico
    plt.title('Ajuste do Modelo Não Linear vs. Dados Reais')
    # Define o rótulo do eixo x
    plt.xlabel('Distância (milhas)')
    # Define o rótulo do eixo y
    plt.ylabel('Tempo (minutos)')
    # Adiciona a legenda ao gráfico
    plt.legend()
    # Ativa a grade
    plt.grid(True)
    # Exibe o gráfico
    plt.show()

def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    """
    Plota o progresso do treinamento de um modelo em dados normalizados,
    mostrando o ajuste atual em cada época.

    Args:
        epoch: O número da época de treinamento atual.
        loss: O valor da perda (loss) na época atual.
        model: O modelo que está sendo treinado.
        distances_norm: Os dados de entrada normalizados.
        times_norm: Os dados de destino normalizados.
    """
    # Limpa o gráfico anterior da célula de saída
    clear_output(wait=True)

    # Faz previsões usando o estado atual do modelo
    predicted_norm = model(distances_norm)

    # Converte tensores para arrays NumPy para plotagem
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()
    
    # Destaca as previsões do gráfico de computação e converte para NumPy
    y_pred_plot = predicted_norm.detach().numpy()

    # Ordena os dados com base na distância para garantir uma linha suave no gráfico
    sorted_indices = x_plot.argsort(axis=0).flatten()

    # Cria uma nova figura para o gráfico
    plt.figure(figsize=(8, 6))

    # Plota os pontos de dados originais normalizados
    plt.plot(x_plot, y_plot, color='orange', marker='o', linestyle='none', label='Dados Reais Normalizados')

    # Plota as previsões do modelo como uma linha
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices], color='green', label='Previsões do Modelo')

    # Define o título do gráfico, incluindo a época atual
    plt.title(f'Época: {epoch + 1} | Progresso do Treinamento Normalizado')
    # Define o rótulo do eixo x
    plt.xlabel('Distância Normalizada')
    # Define o rótulo do eixo y
    plt.ylabel('Tempo Normalizado')
    # Exibe a legenda
    plt.legend()
    # Adiciona uma grade ao gráfico
    plt.grid(True)
    # Mostra o gráfico
    plt.show()

    # Pausa brevemente para permitir que o gráfico seja renderizado
    time.sleep(0.05)