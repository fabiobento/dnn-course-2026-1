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
    # Define o modelo para o modo de avaliação
    model.eval()

    # Desativa o cálculo de gradiente para inferência eficiente
    with torch.no_grad():
        # Faz previsões usando o modelo treinado
        predicted_times = model(distances)

    # Cria uma nova figura para o gráfico
    plt.figure(figsize=(8, 6))
    
    # Plota os pontos de dados reais
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='None', label='Tempos de Entrega Reais')
    
    # Plota a linha prevista pelo modelo
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', marker='None', label='Linha Prevista')
    
    # Define o título do gráfico
    plt.title('Tempos de Entrega: Real vs. Previsto')
    # Define o rótulo do eixo x
    plt.xlabel('Distância (milhas)')
    # Define o rótulo do eixo y
    plt.ylabel('Tempo (minutos)')
    # Exibe a legenda
    plt.legend()
    # Adiciona uma grade ao gráfico
    plt.grid(True)
    # Mostra o gráfico
    plt.show()

    

def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compara e plota as previsões de um modelo em relação a novos dados não lineares.

    Argumentos:
        model: O modelo treinado a ser avaliado.
        new_distances: Os novos dados de entrada para gerar previsões.
        new_times: Os valores alvo reais para comparação.
    """
    # Define o modelo para o modo de avaliação
    model.eval()
    
    # Desativa o cálculo de gradiente para inferência
    with torch.no_grad():
        # Gera previsões usando o modelo
        predictions = model(new_distances)

    # Cria uma nova figura para o gráfico
    plt.figure(figsize=(8, 6))
    
    # Plota os pontos de dados reais
    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange', marker='o', linestyle='None', label='Dados Reais (Bicicletas e Carros)')
    
    # Plota as previsões do modelo
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green', marker='None', label='Previsões do Modelo Linear')
    
    # Define o título do gráfico
    plt.title('Modelo Linear vs. Realidade Não Linear')
    # Define o rótulo do eixo x
    plt.xlabel('Distância (milhas)')
    # Define o rótulo do eixo y
    plt.ylabel('Tempo (minutos)')
    # Adiciona uma legenda ao gráfico
    plt.legend()
    # Adiciona uma grade ao gráfico para melhor legibilidade
    plt.grid(True)
    # Mostra o gráfico
    plt.show()