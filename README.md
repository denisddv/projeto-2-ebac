Previsão de Renda
Descrição

Este projeto realiza uma análise exploratória e previsão de renda utilizando um modelo de regressão linear. A aplicação foi desenvolvida utilizando a biblioteca Streamlit para facilitar a visualização dos dados e a análise dos resultados.
Estrutura do Projeto

    input: Diretório contendo o arquivo CSV com os dados de entrada (previsao_de_renda.csv).
    projeto_2.py: Script principal que carrega os dados, realiza a análise e exibe os resultados na interface do Streamlit.

Requisitos

    Python 3.7 ou superior
    Bibliotecas Python:
        pandas
        seaborn
        matplotlib
        streamlit
        scikit-learn

Instalação

    Clone o repositório:

bash

git clone https://github.com/seu_usuario/previsao_de_renda.git
cd previsao_de_renda

    Crie um ambiente virtual e instale as dependências:

bash

python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
pip install -r requirements.txt

    Coloque o arquivo previsao_de_renda.csv no diretório input.

Uso

Para executar a aplicação Streamlit, utilize o comando:

bash

streamlit run projeto_2.py

A aplicação será aberta em seu navegador padrão, onde você poderá interagir com os gráficos e analisar os resultados do modelo de previsão de renda.
Análise e Resultados
Métricas de Avaliação do Modelo

    Mean Squared Error (MSE): Erro quadrático médio das previsões.
    Mean Absolute Error (MAE): Erro absoluto médio das previsões.
    R² Score: Coeficiente de determinação, indicando a qualidade do ajuste do modelo.

Gráfico de Resíduos

O gráfico de resíduos mostra os erros das previsões em relação aos valores reais de renda.
Gráficos Exploratórios

A aplicação também exibe gráficos exploratórios das variáveis presentes no conjunto de dados, incluindo distribuições e relações bivariadas.
Contribuições

Video do Streamlit

https://github.com/user-attachments/assets/574d894e-da53-4fd2-960e-231ab7bcf7c7



Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias e correções.
