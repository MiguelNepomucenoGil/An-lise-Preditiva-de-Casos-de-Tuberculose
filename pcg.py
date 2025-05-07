import pandas as pd 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt





colunas = st.sidebar.selectbox("Escolha sua vizualização",["TABELA BRUTA","TABELA LIMPA","INDÍCE DE RISCO","Previsão com Machine Learning (ML)"])



#--------------------------------------------------------------------------------- Dataset Bruto
if colunas == "TABELA BRUTA":

    dados_brutos = pd.read_csv("dados_tuberculose.csv",sep=";")
    st.header("Tabela DataSet com dados brutos")
    st.dataframe(dados_brutos)


    st.header("🧾 LEGENDA")
    st.markdown("""

    | Variável         | Descrição                                              | Variável         | Descrição                                              |
    |------------------|--------------------------------------------------------|------------------|--------------------------------------------------------|
    | **ID_AGRAVO**     | ID do Agravo                                           | **DT_NOTIFIC**    | Data de Notificação                                    |
    | **NU_ANO**        | Ano de Notificação                                     | **ID_MUNICIP**    | Município de Notificação                               |
    | **ID_REGIONA**    | Regional de Notificação                                | **ID_UNIDADE**    | Unidade de Notificação                                 |
    | **DT_DIAG**       | Data de Diagnóstico                                    | **NU_IDADE_N**    | Idade                                                  |
    | **CS_SEXO**       | Sexo                                                   | **CS_GESTANT**    | Gestante                                               |
    | **CS_RACA**       | Raça/Cor                                               | **CS_ESCOL_N**    | Escolaridade                                           |
    | **ID_MN_RESI**    | Município de Residência                                | **ID_RG_RESI**    | Regional de Residência                                 |
    | **CS_ZONA**       | Zona de residência do paciente                         | **AGRAVAIDS**     | Doenças/Agravos associados: Aids                       |
    | **AGRAVALCOO**    | Doenças/Agravos associados: Alcoolismo                | **AGRAVDIABE**    | Doenças/Agravos associados: Diabetes                   |
    | **AGRAVDOENC**    | Doenças/Agravos associados: Doença Mental              | **AGRAVOUTRA**    | Doenças/Agravos associados: Outras                     |
    | **AGRAVDROGA**    | Doenças/Agravos associados: Drogas ilícitas            | **AGRAVTABAC**    | Doenças/Agravos associados: Tabagismo                  |
    | **TRATAMENTO**    | Tipo de Entrada                                        | **CULTURA_ES**    | Resultado da cultura de escarro                        |
    | **HIV**           | Resultado da sorologia para HIV                        | **HISTOPATOL**    | Resultado do exame histopatológico                     |
    | **DT_INIC_TR**    | Data de início do tratamento                           | **BACILOSC_1**    | Baciloscopia – 1º mês                                  |
    | **BACILOSC_2**    | Baciloscopia – 2º mês                                  | **BACILOSC_3**    | Baciloscopia – 3º mês                                  |
    | **BACILOSC_4**    | Baciloscopia – 4º mês                                  | **BACILOSC_5**    | Baciloscopia – 5º mês                                  |
    | **BACILOSC_6**    | Baciloscopia – 6º mês                                  | **TRATSUP_AT**    | Tratamento Diretamente Observado (TDO)                |
    | **SITUA_ENCE**    | Situação de encerramento                               | **DT_ENCERRA**    | Data de encerramento                                   |
    | **POP_LIBER**     | População privada de liberdade                         | **TEST_MOLEC**    | Teste Molecular Rápido para TB                         |
    | **TEST_SENSI**    | Teste de Sensibilidade                                 | **RAIOX_TORA**    | Radiografia do Tórax                                   |
    | **FORMA**         | Forma clínica da tuberculose                           |                  |                                                        |
    """)

#---------------------------------------------------------------------------------- DataSet Limpo

if colunas == "TABELA LIMPA":
    dados_tuberculose = pd.read_csv("dados_tuberculose.csv", sep=";")
    dados_tuberculose = dados_tuberculose.drop(columns=["ID_UNIDADE","CS_GESTANT","CS_ESCOL_N","ID_MN_RESI","ID_RG_RESI","HIV","AGRAVOUTRA","CULTURA_ES","DT_NOTIFIC",
                                                        "CS_RACA","TRATAMENTO","POP_LIBER","TEST_SENSI","TRATSUP_AT","TEST_MOLEC","ID_AGRAVO"])

    dados_tuberculose = dados_tuberculose.rename(columns={
        "ID_AGRAVO":"ID",
        "NU_ANO":"ANO",
        "ID_MUNICIP":"MUNICIPIO",
        "ID_REGIONA":"REGIAO",
        "DT_DIAG":"DATA DIAGNOSTICO",
        "NU_IDADE_N":"IDADE",
        "CS_SEXO":"SEXO",
        "CS_ZONA":"ZONA",
        "AGRAVAIDS":"AGRAV HIV",
        "AGRAVALCOO":"AGRAV ALCOOLISMO",
        "AGRAVDIABE":"AGRAV DIABETES",
        "AGRAVDOENC":"AGRAV DOENCA",
        "AGRAVDROGA":"AGRAV DROGAS",
        "AGRAVTABAC":"AGRAV TABACO",
        "DT_INIC_TR":"INICIO DO TRATAMENTO",
        "SITUA_ENCE":"STATUS ENCERRAMENTO",
        "DT_ENCERRA":"DATA ENCERRAMENTO",
        "BACILOSC_1":"1º BACILOSCOPIA",
        "BACILOSC_2":"2º BACILOSCOPIA",
        "BACILOSC_3":"3º BACILOSCOPIA",
        "BACILOSC_4":"4º BACILOSCOPIA",
        "BACILOSC_5":"5º BACILOSCOPIA",
        "BACILOSC_6":"6º BACILOSCOPIA",
        "RAIOX_TORA":"RAIO-X",
        "FORMA":"TIPO"
    })

    dados_tuberculose.index.name = "ID"
    dados_tuberculose = dados_tuberculose.dropna()
    dados_tuberculose = dados_tuberculose[~dados_tuberculose.apply(lambda row: row.astype(str).str.contains('ignorado', case=False).any(), axis=1)]
    dados_tuberculose['BACILOSCOPIA_NEGATIVA'] = dados_tuberculose[['1º BACILOSCOPIA', '2º BACILOSCOPIA', '3º BACILOSCOPIA', '4º BACILOSCOPIA', '5º BACILOSCOPIA', '6º BACILOSCOPIA']].apply(lambda row: sum(row.str.contains('Negativa', case=False)), axis=1)




    st.title("Tabela DataSet limpo")
    st.write("Essa tabela foi feita a partir de Machine Learning, com o objetivo de prever casos de tuberculose e seus agravantes.")

    st.dataframe(dados_tuberculose, use_container_width=True)

 
    st.header("🧾LEGENDA")
    st.markdown("""
    ###  

    | Variável                  | Descrição                                               |
    |---------------------------|---------------------------------------------------------|
    | **ID**                    | ID do Agravo                                            |
    | **ANO**                   | Ano de Notificação                                      |
    | **MUNICIPIO**             | Município de Notificação                                |
    | **REGIAO**                | Regional de Notificação                                 |
    | **DATA DIAGNOSTICO**      | Data de Diagnóstico                                     |
    | **IDADE**                 | Idade                                                   |
    | **SEXO**                  | Sexo                                                    |
    | **ZONA**                  | Zona de residência do paciente                          |
    | **AGRAV HIV**             | Doenças e agravos associados à Aids                    |
    | **AGRAV ALCOOLISMO**      | Doenças e agravos associados ao Alcoolismo             |
    | **AGRAV DIABETES**        | Doenças e agravos associados ao Diabetes               |
    | **AGRAV DOENCA**          | Doenças e agravos associados à Doença Mental           |
    | **AGRAV DROGAS**          | Doenças e agravos associados ao uso de drogas ilícitas |
    | **AGRAV TABACO**          | Doenças e agravos associados ao Tabagismo              |
    | **INICIO DO TRATAMENTO**  | Data em que o paciente iniciou o tratamento atual       |
    | **STATUS ENCERRAMENTO**   | Situação de encerramento                                |
    | **DATA ENCERRAMENTO**     | Data de encerramento                                    |
    | **1º BACILOSCOPIA**       | Baciloscopia no 1º mês                                  |
    | **2º BACILOSCOPIA**       | Baciloscopia no 2º mês                                  |
    | **3º BACILOSCOPIA**       | Baciloscopia no 3º mês                                  |
    | **4º BACILOSCOPIA**       | Baciloscopia no 4º mês                                  |
    | **5º BACILOSCOPIA**       | Baciloscopia no 5º mês                                  |
    | **6º BACILOSCOPIA**       | Baciloscopia no 6º mês                                  |
    | **RAIO-X**                | Radiografia do tórax                                    |
    | **TIPO**                  | Forma clínica da tuberculose                            |
    """)



#------------------------------------------------------------------- Indice de Risco

if colunas == "INDÍCE DE RISCO":
    
    dados_tuberculose = pd.read_csv("dados_tuberculose.csv", sep=";")
    dados_tuberculose = dados_tuberculose.drop(columns=["ID_UNIDADE","CS_GESTANT","CS_ESCOL_N","ID_MN_RESI","ID_RG_RESI","HIV","AGRAVOUTRA","CULTURA_ES","DT_NOTIFIC",
                                                        "CS_RACA","TRATAMENTO","POP_LIBER","TEST_SENSI","TRATSUP_AT","TEST_MOLEC","ID_AGRAVO"])

    dados_tuberculose = dados_tuberculose.rename(columns={
        "ID_AGRAVO":"ID",
        "NU_ANO":"ANO",
        "ID_MUNICIP":"MUNICIPIO",
        "ID_REGIONA":"REGIAO",
        "DT_DIAG":"DATA DIAGNOSTICO",
        "NU_IDADE_N":"IDADE",
        "CS_SEXO":"SEXO",
        "CS_ZONA":"ZONA",
        "AGRAVAIDS":"AGRAV HIV",
        "AGRAVALCOO":"AGRAV ALCOOLISMO",
        "AGRAVDIABE":"AGRAV DIABETES",
        "AGRAVDOENC":"AGRAV DOENCA",
        "AGRAVDROGA":"AGRAV DROGAS",
        "AGRAVTABAC":"AGRAV TABACO",
        "DT_INIC_TR":"INICIO DO TRATAMENTO",
        "SITUA_ENCE":"STATUS ENCERRAMENTO",
        "DT_ENCERRA":"DATA ENCERRAMENTO",
        "BACILOSC_1":"1º BACILOSCOPIA",
        "BACILOSC_2":"2º BACILOSCOPIA",
        "BACILOSC_3":"3º BACILOSCOPIA",
        "BACILOSC_4":"4º BACILOSCOPIA",
        "BACILOSC_5":"5º BACILOSCOPIA",
        "BACILOSC_6":"6º BACILOSCOPIA",
        "RAIOX_TORA":"RAIO-X",
        "FORMA":"TIPO"
    })

    dados_tuberculose.index.name = "ID"
    dados_tuberculose = dados_tuberculose.dropna()
    dados_tuberculose = dados_tuberculose[~dados_tuberculose.apply(lambda row: row.astype(str).str.contains('ignorado', case=False).any(), axis=1)]
    dados_tuberculose['BACILOSCOPIA_NEGATIVA'] = dados_tuberculose[['1º BACILOSCOPIA', '2º BACILOSCOPIA', '3º BACILOSCOPIA', '4º BACILOSCOPIA', '5º BACILOSCOPIA', '6º BACILOSCOPIA']].apply(lambda row: sum(row.str.contains('Negativa', case=False)), axis=1)
    
    st.title("Tabela de Pontuação de Risco")
    st.write("Esta tabela foi gerada a partir de um modelo de Machine Learning e tem como objetivo estimar o índice de risco dos pacientes com base em seus diferentes fatores agravantes.")

    pesos = {
        'AGRAV HIV': 5,
        'AGRAV DIABETES': 2.5,
        'AGRAV DROGAS': 2,
        'AGRAV ALCOOLISMO': 1.5,
        'AGRAV TABACO': 1
    }

    def calcular_pontuacao(row):
        pontuacao = 0
        for agravante, peso in pesos.items():
            if row[agravante] == 'Sim':
                pontuacao += peso
        return pontuacao

    def determinar_nivel_risco(pontuacao):
        if pontuacao <= 3:
            return 'Baixo Risco'
        elif pontuacao <= 6:
            return 'Médio Risco'
        else:
            return 'Alto Risco'

    dados_tuberculose['PONTUACAO RISCO'] = dados_tuberculose.apply(calcular_pontuacao, axis=1)
    dados_tuberculose['NÍVEL DE RISCO'] = dados_tuberculose['PONTUACAO RISCO'].apply(determinar_nivel_risco)

    tabela_pontuacao = dados_tuberculose[['MUNICIPIO', 'IDADE', 'SEXO', 'AGRAV HIV','AGRAV DIABETES','AGRAV DROGAS','AGRAV ALCOOLISMO','AGRAV TABACO','PONTUACAO RISCO', 'NÍVEL DE RISCO','STATUS ENCERRAMENTO']]

    st.dataframe(tabela_pontuacao, use_container_width=True)

  
    st.markdown("""
    ### 🧮 **Pontos por Agravante**

    | Agravante           | Pontuação |
    |---------------------|:---------:|
    | 🦠 **HIV**           | **5**     |
    | 🍬 **Diabetes**      | **2.5**   |
    | 💉 **Drogas**        | **2**     |
    | 🍺 **Alcoolismo**    | **1.5**   |
    | 🚬 **Tabaco**        | **1**     |
    """)

    st.markdown("""
    ### 🗂️ **Legenda – Nível de Risco para Tuberculose**

    - 🟢 **Baixo Risco**  
     **Pontuação:** 0 a 3  
     **Descrição:** Poucos ou nenhum fator agravante relevante. Baixa probabilidade de desenvolver tuberculose ativa.

    - 🟡 **Médio Risco**  
     **Pontuação:** 3.5 a 6  
     **Descrição:** Presença de fatores agravantes moderados. Exige atenção e acompanhamento preventivo.

    - 🔴 **Alto Risco**  
     **Pontuação:** Acima de 6  
     **Descrição:** Fatores altamente agravantes, como HIV. Alta probabilidade de desenvolver tuberculose ativa. Recomendado acompanhamento médico intensivo.
    """)
    
#------------------------------------------------------------------------------------- Previsao com ML

elif colunas == "Previsão com Machine Learning (ML)":
    st.title("🤖 Previsão do Status de Encerramento com Machine Learning")


    dados_tuberculose = pd.read_csv("dados_tuberculose.csv", sep=";")
    dados_tuberculose = dados_tuberculose.drop(columns=["ID_UNIDADE", "CS_GESTANT", "CS_ESCOL_N", "ID_MN_RESI", "ID_RG_RESI", "HIV", "AGRAVOUTRA", "CULTURA_ES", 
                                                        "DT_NOTIFIC", "CS_RACA", "TRATAMENTO", "POP_LIBER", "TEST_SENSI", "TRATSUP_AT", "TEST_MOLEC", "ID_AGRAVO"])

    dados_tuberculose = dados_tuberculose.rename(columns={
        "ID_AGRAVO":"ID",
        "NU_ANO":"ANO",
        "ID_MUNICIP":"MUNICIPIO",
        "ID_REGIONA":"REGIAO",
        "DT_DIAG":"DATA DIAGNOSTICO",
        "NU_IDADE_N":"IDADE",
        "CS_SEXO":"SEXO",
        "CS_ZONA":"ZONA",
        "AGRAVAIDS":"AGRAV HIV",
        "AGRAVALCOO":"AGRAV ALCOOLISMO",
        "AGRAVDIABE":"AGRAV DIABETES",
        "AGRAVDOENC":"AGRAV DOENCA",
        "AGRAVDROGA":"AGRAV DROGAS",
        "AGRAVTABAC":"AGRAV TABACO",
        "DT_INIC_TR":"INICIO DO TRATAMENTO",
        "SITUA_ENCE":"STATUS ENCERRAMENTO",
        "DT_ENCERRA":"DATA ENCERRAMENTO",
        "BACILOSC_1":"1º BACILOSCOPIA",
        "BACILOSC_2":"2º BACILOSCOPIA",
        "BACILOSC_3":"3º BACILOSCOPIA",
        "BACILOSC_4":"4º BACILOSCOPIA",
        "BACILOSC_5":"5º BACILOSCOPIA",
        "BACILOSC_6":"6º BACILOSCOPIA",
        "RAIOX_TORA":"RAIO-X",
        "FORMA":"TIPO"
    })

    dados_tuberculose.index.name = "ID"


    colunas_entrada = ['IDADE', 'SEXO', 'ZONA', 'AGRAV HIV', 'AGRAV DIABETES', 'AGRAV DROGAS', 'AGRAV ALCOOLISMO', 'AGRAV TABACO']
    X = dados_tuberculose[colunas_entrada]
    y = dados_tuberculose['STATUS ENCERRAMENTO']

    le_dict = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)


  

   
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y_encoder.classes_, yticklabels=y_encoder.classes_, ax=ax)
    st.pyplot(fig)

    st.subheader("🎯 Prever situação de encerramento de um paciente")

    idade = st.slider("Idade", 0, 120, 30)
    sexo = st.selectbox("Sexo", le_dict['SEXO'].classes_)
    zona = st.selectbox("Zona", le_dict['ZONA'].classes_)
    hiv = st.selectbox("HIV", le_dict['AGRAV HIV'].classes_)
    diabetes = st.selectbox("Diabetes", le_dict['AGRAV DIABETES'].classes_)
    drogas = st.selectbox("Drogas", le_dict['AGRAV DROGAS'].classes_)
    alcool = st.selectbox("Alcoolismo", le_dict['AGRAV ALCOOLISMO'].classes_)
    tabaco = st.selectbox("Tabagismo", le_dict['AGRAV TABACO'].classes_)

    input_dict = {
        'IDADE': idade,
        'SEXO': le_dict['SEXO'].transform([sexo])[0],
        'ZONA': le_dict['ZONA'].transform([zona])[0],
        'AGRAV HIV': le_dict['AGRAV HIV'].transform([hiv])[0],
        'AGRAV DIABETES': le_dict['AGRAV DIABETES'].transform([diabetes])[0],
        'AGRAV DROGAS': le_dict['AGRAV DROGAS'].transform([drogas])[0],
        'AGRAV ALCOOLISMO': le_dict['AGRAV ALCOOLISMO'].transform([alcool])[0],
        'AGRAV TABACO': le_dict['AGRAV TABACO'].transform([tabaco])[0]
    }

    entrada = pd.DataFrame([input_dict])

    if st.button("Prever Status de Encerramento"):
        predicao = modelo.predict(entrada)[0]
        classe_predita = y_encoder.inverse_transform([predicao])[0]
        st.success(f"🧾 Previsão: {classe_predita}")


  
