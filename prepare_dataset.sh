#!/bin/bash

# Verifica se os parâmetros foram passados
if [ $# -ne 2 ]; then
  echo "Uso: $0 <diretório_origem> <diretório_destino>"
  exit 1
fi

# Atribui os parâmetros a variáveis
diretorio_origem="$1"
diretorio_destino="$2"

# Verifica se o diretório de origem existe
if [ ! -d "$diretorio_origem" ]; then
  echo "O diretório de origem '$diretorio_origem' não existe."
  exit 1
fi

# Cria o diretório de destino, se não existir
mkdir -p "$diretorio_destino"

# Copia os arquivos dos subdiretórios para o diretório de destino
for dir in "$diretorio_origem"/*; do
  if [ -d "$dir" ]; then
    # Encontra e copia todos os arquivos de cada subdiretório
    find "$dir" -type f -exec cp {} "$diretorio_destino" \;
  fi
done

echo "Arquivos copiados para '$diretorio_destino'."
