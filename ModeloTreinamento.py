#cu
import pandas as pd
import joblib 
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

#ETAPA 01 - CARREGAR ARQUIVOS
def CarregarDados(caminho_do_arquivo = "ModeloTreinamento.py"):
    try:
        if os.path.exists(caminho_do_arquivo):
            df =pd.read.read_cdv(caminho_do_arquivo, enconding="latin1",sp=',')
            print("O arquivo foi carregado corretamente")

            return df
        else:
            print('O arquivo não foi encontrado')
            return None
    
    except Exception as e:
        print("Erro inesperado ao procurar o arquivo")
        return None
    