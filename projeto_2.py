import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set(context='talk', style='ticks')

st.set_page_config(
    page_title="Previsão de Renda",
    page_icon=":bar_chart:",
    layout="wide",
)

st.write('# Análise e Previsão de Renda')

# Carregar o dataset
renda = pd.read_csv('./input/previsao_de_renda.csv')

# Seleção das variáveis para visualizar a distribuição
features = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']

# Verificar a presença de valores faltantes nas variáveis selecionadas
renda = renda.dropna(subset=features)

# Lista de variáveis categóricas para criar variáveis dummy
categorical_features = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']

# Separar as variáveis independentes (X) e a variável alvo (y), excluindo 'data_ref'
X = renda.drop(columns=['renda', 'data_ref'])
y = renda['renda']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir o transformador de colunas com OneHotEncoder para variáveis categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # Manter as colunas restantes como estão
)

# Definir o pipeline com pré-processamento e o modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer previsões
y_pred = pipeline.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calcular resíduos
residuals = y_test - y_pred

# Métricas de Avaliação
st.write('## Métricas de Avaliação do Modelo')
st.write(f'**Mean Squared Error (MSE):** {mse:.2f}')
st.write(f'**Mean Absolute Error (MAE):** {mae:.2f}')
st.write(f'**R² Score:** {r2:.2f}')

# Gráfico de Resíduos
st.write('## Gráfico de Resíduos')
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Reais de Renda')
plt.ylabel('Resíduos (Erros)')
plt.title('Gráfico de Resíduos do Modelo com Dummies')
st.pyplot(plt)

# Plots existentes no arquivo original
fig, ax = plt.subplots(8, 1, figsize=(10, 70))
renda[['posse_de_imovel', 'renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref', y='renda', hue='qtd_filhos', data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref', y='renda', hue='educacao', data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref', y='renda', hue='tipo_residencia', data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(fig)

st.write('## Gráficos Bivariados')
fig, ax = plt.subplots(7, 1, figsize=(10, 50))
sns.barplot(x='posse_de_imovel', y='renda', data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo', y='renda', data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos', y='renda', data=renda, ax=ax[2])
sns.barplot(x='tipo_renda', y='renda', data=renda, ax=ax[3])
sns.barplot(x='educacao', y='renda', data=renda, ax=ax[4])
sns.barplot(x='estado_civil', y='renda', data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia', y='renda', data=renda, ax=ax[6])
sns.despine()
st.pyplot(fig)






