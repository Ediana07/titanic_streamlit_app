# coding: utf-8

# # Titanic - Probabilidade de Sobrevivência
# Este notebook cria um app interativo para prever a chance de sobrevivência de um passageiro do Titanic, utilizando o dataset do Kaggle.

# In[1]:


#get_ipython().system('pip install pandas scikit-learn streamlit')


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.header('🚢 Titanic - Probabilidade de Sobrevivência')

st.markdown("""
Este app prevê a **probabilidade de sobrevivência** de um passageiro no Titanic.
Quer saber suas chances de sobrevivência? Preencha seus dados abaixo.
""")

# Carregando os dados
@st.cache_data # Guarda em cache o resultado da função
def load_data():
    data = pd.read_csv('titanic.csv')
    return data

df = load_data() #Define uma função, ou melhor, um bloco de código que pode ser reutilizado quando precisar carregar dados

# Selecionando variáveis relevantes, removendo as linhas que contêm valores ausentes
df = df[['survived', 'pclass', 'sex', 'age']].dropna()

# Codificando variável categórica
le = LabelEncoder() #criando um codificador de rótulos convertendo categorias de texto em números inteiros
df['sex'] = le.fit_transform(df['sex']) # para "0" e "1"

# Definindo X e y
X = df[['pclass', 'sex', 'age']] # variáveis preditoras
y = df['survived'] # variável alvo definida como categórica

# Treinando o modelo
model = LogisticRegression()
model.fit(X, y)

# Interface do usuário
st.sidebar.header('Informe seus dados:') # Colocando em negrito, organizando o campo de entrada

pclass = st.sidebar.selectbox('Classe do Bilhete', (1, 2, 3)) # criando uma caixa de seleção para o usuário escolher a classe da passagem
sex = st.sidebar.selectbox('Sexo', ('Masculino', 'Feminino')) # Outra caixa para informação do sexo
age = st.sidebar.slider('Idade', 0, 100, 25) # Criando um controle deslizante (slider) para o usuário indicar a idade (min=0, max=100, cursor começa no 25) 

# Codificando sexo
sex_encoded = 1 if sex == 'male' else 0 #expressão condicional/tenária. Se for masc = 1, caso contrário será 0 (fem)

# Fazendo previsão
input_data = np.array([[pclass, sex_encoded, age]]) #criando uma matriz
prob = model.predict_proba(input_data)[0][1] #retornando uma array com as probalidades de cada classe

# Mostrando resultado
st.subheader('Resultado:')
st.write(f'**Probabilidade de sobreviver:** {prob*100:.2f}%')

if prob >= 0.5: #verifica se a probabilidade é de 50%
    st.success('Alta chance de sobreviver! 🎉') #função para mostrar a mensagem em verde
else:
    st.error('Baixa chance de sobreviver... 😢')  #função para mostrar a mensagem em verde

# Mostrando dados
with st.expander('🔍 Ver dados utilizados no treinamento'): #função para criar caixa que expande
    st.dataframe(df)
