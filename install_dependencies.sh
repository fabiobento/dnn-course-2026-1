#!/bin/bash

# Para a execução em caso de erro
set -e

echo "Preparando a instalação das dependências..."

# Cria um ambiente virtual chamado "venv" se ele não existir
if [ ! -d "venv" ]; then
    echo "Criando o ambiente virtual 'venv'..."
    python3 -m venv venv
fi

# Ativa o ambiente virtual
echo "Ativando o ambiente virtual..."
source venv/bin/activate

# Atualizando o pip dentro do ambiente virtual
echo "Atualizando o pip..."
pip install --upgrade pip

# Instalando dependências
echo "Instalando os pacotes necessários..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Instalação concluída com sucesso no ambiente virtual!"
echo "Para abrir os notebooks, você precisa ATIVAR o ambiente"
echo "virtual primeiro. Para fazer isso, execute:"
echo ""
echo "    source venv/bin/activate"
echo "    jupyter notebook"
echo "============================================================"
