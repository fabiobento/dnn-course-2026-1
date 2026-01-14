# A regra de aprendizagem do perceptron e as atualizações de peso

## Introdução
Imagine um modelo simples, mas poderoso, que consegue classificar dados com precisão. O notebook é o **Perceptron**, um modelo fundamental nas redes neurais. Ele utiliza a Regra de Aprendizado do Perceptron (**Perceptron Learning Rule**) para aprender com seus erros passo a passo. Esta leitura mostrará como esse processo básico ajusta os pesos (**weights**) e o viés (**bias**) para classificar corretamente dados que podem ser separados por uma linha reta.


## Entendendo a Regra de Aprendizado do Perceptron
O **Perceptron** é o tipo mais básico de rede neural. Ele foi projetado para dividir dados em dois grupos (classificação binária). Pense nele como tendo:
* Um neurônio (ou nó)
* Vários pesos ajustáveis
* Um viés (**bias**)

Ele faz suposições somando suas entradas multiplicadas por seus respectivos pesos e, em seguida, aplicando uma função de degrau (**step function**). Se um palpite estiver errado, a Regra de Aprendizado do Perceptron diz a ele como alterar seus pesos e viés para melhorar.

## Mecânica do Perceptron: Como Ele Aprende
As etapas para a Regra de Aprendizado do Perceptron são simples, mas eficazes:

1.  **Início:** Comece com pequenos pesos aleatórios e um viés (ou defina todos como zero).
2.  **Palpite:** Para cada dado de entrada, calcule um total multiplicando cada entrada por seu peso e somando o viés. Em seguida, use uma **step function** para fazer uma previsão final (geralmente 0 ou 1).
3.  **Encontrar o Erro:** Compare o palpite do **Perceptron** com a resposta correta real.
4.  **Atualizar Pesos e Viés:** Se houver um erro, ajuste os pesos e o viés. Veja como:
    * Para cada peso $w_i$ conectado à entrada $x_i$: $w_i \rightarrow w_i + \Delta w_i$
    * Para o viés $b$: $b \rightarrow b + \Delta b$ onde $\Delta b = \eta \cdot (t - o)$
5.  **Repetir:** Passe por essas etapas para muitos pontos de dados e por várias "épocas" (uma passagem completa por todos os dados de treinamento). Isso continua até que o **Perceptron** não cometa mais erros, o que significa que ele aprendeu a separar os dados.

## Convergência: Alcançando uma Solução
Uma grande vantagem do **Perceptron** é que ele tem garantia de encontrar uma solução para dados linearmente separáveis. Isso significa que, se você puder desenhar uma linha reta (ou um plano plano em dimensões superiores) para separar perfeitamente seus dados, o **Perceptron** eventualmente encontrará essa linha e classificará corretamente todos os seus dados de treinamento sem erros. Isso funciona desde que você escolha uma boa taxa de aprendizado (**learning rate**).

## Desafios: Além da Linearidade
Mesmo com seus pontos fortes, o **Perceptron** tem uma limitação: ele só pode resolver problemas onde os dados são linearmente separáveis. Ele não consegue lidar com dados que não podem ser divididos por uma única linha reta, como o famoso problema **XOR**. Isso mostra por que precisamos de modelos mais avançados, como redes neurais de múltiplas camadas, que podem aprender padrões mais complexos.


Clique em uma dos links abaixo para abrir o notebook:
<table>
  <td>
    <a href="https://colab.research.google.com/github/fabiobento/dnn-course-2026-1/blob/main/aulas/dl-pytorch/intro-dl/docs/perceptron-learn.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  </td>
  <td>
    <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/fabiobento/dnn-course-2026-1/blob/main/aulas/dl-pytorch/intro-dl/docs/perceptron-learn.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
  </td>
</table>

## Visualização: Dando Vida aos Conceitos
Ver a Regra de Aprendizado do Perceptron em ação pode realmente ajudar. Vamos dar uma olhada em um exemplo de código.

||
|---|
|Próximo item **[➡️Treinar o Perceptron com a Regra de Aprendizagem do Perceptron](./perceptron-learn-train.md)**|
|Item anterior **[⬅️Desenvolvendo o forward pass do perceptron em Pytorch](./perceptron-pytorch.md)**|
||
|Página inicial do curso **[⬆️Introdução ao Deep Learning com PyTorch](./intro-dl.md)**|

