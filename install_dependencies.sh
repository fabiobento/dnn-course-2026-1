#!/bin/bash

# Para a execução em caso de erro
set -e

echo "Preparando a instalação das dependências..."

# 1. Cria um ambiente virtual chamado "venv" se ele não existir
if [ ! -d "venv" ]; then
    echo "Criando o ambiente virtual 'venv'..."
    python3 -m venv venv
fi

# 2. Ativa o ambiente virtual
echo "Ativando o ambiente virtual..."
source venv/bin/activate

# 3. Atualizando o pip dentro do ambiente virtual
echo "Atualizando o pip..."
pip install --upgrade pip

# 4. Instalando dependências
if [ -f "requirements.txt" ]; then
    echo "Instalando os pacotes necessários do requirements.txt..."
    pip install -r requirements.txt
else
    echo "ERRO: Arquivo requirements.txt não encontrado!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Instalação concluída com sucesso no ambiente virtual!"
echo "As bibliotecas PyTorch, Torchvision e utilitários estão prontas."
echo ""
echo "Para abrir os notebooks, você precisa ATIVAR o ambiente"
echo "virtual primeiro. Para fazer isso, execute:"
echo ""
echo "    source venv/bin/activate"
echo "    jupyter notebook"
echo "============================================================"