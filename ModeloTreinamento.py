import pandas as pd
import joblib 
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

#ETAPA 01 - CARREGAR ARQUIVOS
def CarregarDados(caminho_do_arquivo="historicoAcademico.csv"):
    try:
        if os.path.exists(caminho_do_arquivo):
            df = pd.read_csv(caminho_do_arquivo, encoding="latin1", sep=',')
            print("O arquivo foi carregado corretamente")
            return df
        else:
            print("O arquivo não foi encontrado")
            return None
    except Exception as e:
        print(f"Erro inesperado ao procurar o arquivo: {e}")
        return None

dados=CarregarDados()
print(dados)

