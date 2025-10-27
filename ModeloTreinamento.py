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
#======função armazenamento======#

dados=CarregarDados()

#======Etapa - Divisão de Dados======#

#definição de x (features) e y (target)
if dados is not None:
    print(f"\nRegistro carregados: {len(dados)}") 
    print("\nIniciando o pipeline de Treinamento")

    TARGET_COLUM = "Status_Final"

    try:
        X = dados.drop(TARGET_COLUM, axis=1)
        Y= dados[TARGET_COLUM]
        print(f"\nColuna X definida como: {list(X.columns)}")
        print(f"Coluna Y definida como: {TARGET_COLUM}\n")

    except:
        print("\n === ERRO CRITICO ===")
        print(f"Não foi possivel identificar {TARGET_COLUM}")
        print(f"Colunas disponiveis {list(dados.columns)}")
        print(f'Atualize o valor de {TARGET_COLUM}')

        #encerrar o script#
        exit()
    #======Etapa 2.2 Divisão entre treino e teste======#
    print(f'Dividindo dados entre treino e teste')
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y,
        test_size=0.2,       #Garante que 20% serão utilizado para TESTE; e o restante para TREINAMENTO DO MODELO
        random_state=42,    #Garantir a reprodutividade dos dados
        stratify=Y          #Mantem a proproção dos dados finais em codificação binaria
    )
    print(f"Dados de Treinamento: {len(X_train)} | Dados de Teste: {len(X_test)}")

    #Criação da pipeline de machine learnig

    print("Criação da pipeline de machine learnig")
    #scaler = normaliza dos dados (td na mesma escala)
    #model = aplica o modelo de regressão logistica
    pipeline_model = pipeline.Pipeline([
        ('scaler',preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    #Etapa 4 treinamento e avaliação dos dados
    print('====== TREINAMENTO DO MODELO ======')
    pipeline_model.fit(X_train, Y_train)
    print(f' Modelo treinado. AVALIANDO OS DADOS TREINADOS')
    Y_pred = pipeline_model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    report = metrics.classification_report(Y_test, Y_pred)
    print(" ====== RELATÓRIO DE AVALIAÇÃO GERAL ======")
    print(F'Acuracia geral: {accuracy * 100:2}%')
    print(f"Relatório de classificação detalhado:")
    print(report)

    #RETA FINAL SALVANDO O MODELO

    model_filename = 'ModeloDePrevisãoDeDesempenho.joblib'
    print(f"\nSalvando o pipeline treinado em {model_filename}")
    joblib.dump(pipeline_model, model_filename)
    print("PROCESSO CONCLUIDO")
    print(f"O Arquivo {model_filename} está pronto para uso")
          
else:
    print("O pipeline não pode continuar, pois o dados não forma carregados")
