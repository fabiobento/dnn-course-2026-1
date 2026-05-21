## Gerenciamento de Dados com PyTorch - Referências

### Aprendizado a partir de dados (Learning from data)

* **MIT CSAIL · Data-Centric AI (notas de aula/visão geral):** Contrapõe os fluxos de trabalho centrados no modelo (*model-centric*) versus centrados nos dados (*data-centric*) e apresenta um roteiro prático para melhorar a qualidade dos dados por meio de iteração, padrões de rotulagem e análise direcionada de erros. [Introduction to Data-Centric AI](https://dcai.csail.mit.edu/2024/data-centric-model-centric)
* **CACM: The Principles of Data-Centric AI:** Um conjunto citável de princípios para design de conjuntos de dados, versionamento, consistência de rotulagem, aumento de dados e avaliação que mudam o foco dos modelos para os dados. [cacm.acm.org](https://cacm.acm.org/research/the-principles-of-data-centric-ai)
* **Pesquisa ArXiv: Data-Centric Artificial Intelligence:** Uma taxonomia abrangente e um mapa da literatura cobrindo avaliação, curadoria, aumento de dados, supervisão fraca (*weak supervision*) e governança para ML. [arXiv](https://arxiv.org/html/2212.11854v4)

### Organização de dados, divisões e validação à prova de vazamento (Data organization, splits & leakage-proof validation)

* **Guia de validação cruzada do Scikit-learn:** Explica quando usar KFold, StratifiedKFold, GroupKFold e divisões baseadas em tempo, além das armadilhas comuns de vazamento de dados (*leakage*) a serem evitadas. [scikit-learn.org](https://scikit-learn.org/stable/modules/cross_validation.html)
* **Vazamento de dados em ML (MachineLearningMastery):** Exemplos tabulares concretos de características alvo/vazadas (*target/leaky features*) e como prevenir o vazamento por meio de pipelines e design de validação adequados. [MachineLearningMastery.com](https://www.machinelearningmastery.com/data-leakage-machine-learning)

### Oxford Flowers 102 

* **Página oficial do VGG (Oxford):** Descrição do conjunto de dados, lista de classes, contagem de imagens, divisões oficiais e links de download para o conjunto de flores de 102 categorias. [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* **Cartão do conjunto de dados (Huggingface):** Especificações rápidas, recursos, trecho de código para carregamento e links para *benchmarks*/SOTA para o Flowers-102. [huggingface.com](https://huggingface.co/datasets/Voxel51/OxfordFlowers102)
* **Referência do artigo original (visão geral):** Contexto sobre como o conjunto de dados foi coletado e o protocolo de avaliação utilizado no trabalho original. [ResearchGate](https://www.researchgate.net/publication/221551861_Automated_Flower_Classification_over_a_Large_Number_of_Classes)

### Pré-processamento e pipelines de transformação (Pre-processing & transform pipelines)

* **Notas de aula/slides do CS231n (pré-processamento de dados):** Abordagem intuitiva de subtração da média, padronização, redimensionamento/recorte e por que manter o pré-processamento consistente entre o treino e o teste. [cs231n.github.io](http://cs231n.github.io/) | [cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2023/lecture_7.pdf)
* **Raschka: Feature scaling/normalization (introdução em blog):** Comparação clara e sem foco matemático entre padronização (*standardization*) versus normalização (*normalization*) e quando cada uma é apropriada. [sebastianraschka.com](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
* **Blog da Voxel51: melhores práticas de pré-processamento de imagens:** Orientação prática sobre estratégias de redimensionamento, escolhas de normalização e como lidar com mudanças de domínio (*domain shifts*) com foco em visualização. [Voxel51](https://voxel51.com/blog/image-preprocessing-best-practices-to-optimize-your-ai-workflows)

### Aumento de dados e robustez (Data augmentation & robustness) 

* **Roboflow — What is Data Augmentation? The Ultimate Guide:** Catálogo prático dos aumentos de dados mais comuns em Visão Computacional com dicas de parâmetros, ressalvas e exemplos de casos de uso. [Roboflow Blog](https://blog.roboflow.com/data-augmentation/)


* **Blog do Stanford AI Lab — Automating Data Augmentation:** Visão geral acessível de políticas baseadas em busca (AutoAugment/RandAugment) e suas compensações (*trade-offs*) na prática. [ai.stanford.edu](https://ai.stanford.edu/blog/data-augmentation)
* **YouTube — Albumentations tutorial (focado em PyTorch):** Vídeo passo a passo demonstrando aumentos rápidos de imagem, composição de pipelines e visualização de resultados. [youtube.com](https://www.youtube.com/watch?pp=0gcJCf8Ao7VqN5tD&v=rAdLwKJBvPM)