import torch
import matplotlib.pyplot as plt

def plot_results(model, distances, times):
    """
    Plota os pontos de dados reais e a linha prevista pelo modelo para um determinado conjunto de dados.

    Argumentos:
        model: O modelo de aprendizado de máquina treinado a ser usado para previsões.
        distances: Os pontos de dados de entrada (features) para o modelo.
        times: Os pontos de dados alvo (labels) para o gráfico.
    """
    # Define o modelo para o modo de avaliação (importante para consistência nas previsões)
    model.eval()

    # Desativa o cálculo de gradiente para economizar memória e acelerar a inferência
    with torch.no_grad():
        # Gera as previsões (y_pred) com base nas distâncias fornecidas
        predicted_times = model(distances)

    # Cria a janela do gráfico com um tamanho definido
    plt.figure(figsize=(8, 6))
    
    # Plota os dados reais como pontos (scatter plot manual usando linestyle='None')
    # Convertemos os tensores do PyTorch para arrays NumPy para que o Matplotlib possa processá-los
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='None', label='Tempos de Entrega Reais')
    
    # Plota a linha de tendência que o modelo aprendeu
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', marker='None', label='Linha Prevista')
    
    # Configurações de rótulos e títulos
    plt.title('Tempos de Entrega: Real vs. Previsto')
    plt.xlabel('Distância (milhas)')
    plt.ylabel('Tempo (minutos)')
    
    # Adiciona a legenda baseada nos 'labels' definidos acima
    plt.legend()
    # Ativa as linhas de grade para facilitar a leitura dos valores
    plt.grid(True)
    # Exibe o gráfico finalizado
    plt.show()

def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compara e plota as previsões de um modelo em relação a novos dados não lineares.

    Argumentos:
        model: O modelo treinado a ser avaliado.
        new_distances: Os novos dados de entrada para gerar previsões.
        new_times: Os valores alvo reais para comparação.
    """
    # Coloca o modelo em modo de avaliação
    model.eval()
    
    # Bloco que garante que os pesos do modelo não sejam alterados durante a visualização
    with torch.no_grad():
        # Obtém as previsões do modelo para os novos dados
        predictions = model(new_distances)

    plt.figure(figsize=(8, 6))
    
    # Plota os novos dados que possuem comportamento não linear (ex: diferentes veículos)
    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange', marker='o', linestyle='None', label='Dados Reais (Bicicletas e Carros)')
    
    # Plota a linha de previsão para mostrar como o modelo linear tenta se ajustar a dados curvos
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green', marker='None', label='Previsões do Modelo Linear')
    
    # Títulos e legendas
    plt.title('Modelo Linear vs. Realidade Não Linear')
    plt.xlabel('Distância (milhas)')
    plt.ylabel('Tempo (minutos)')
    
    plt.legend()
    plt.grid(True)
    plt.show()