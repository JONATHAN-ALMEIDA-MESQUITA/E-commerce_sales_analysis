#Importando as principais bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

#Bibliotecas para analise de regressão linear
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Estrutura de dados





#Estrutura do APP
# Título da aplicação
st.title('📈 Análise de Vendas de Comércio Eletrônico')

# Informações gerais sobre o DataFrame
st.subheader('Informações sobre o DataFrame')

st.markdown(''' ###### Conjunto de dados abrangendo uma variedade de categorias de produtos de um E-comerce, preços, avaliações de clientes e tendências de vendas no último ano. 
###### Este conjunto de dados é ideal para analisar tendências de mercado, comportamento do cliente e desempenho de vendas. 
Vamos explorar os dados para descobrir insights que podem otimizar listagens de precos de produtos, estratégias de campanhas de marketing e previsão de vendas!
''')

# Leitura do DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv('data/ecommerce_sales_analysis_input.csv')

    return df

df = load_data()

# Exibição inicial do DataFrame
if st.checkbox('Mostrar dados'):
    st.write(df)



# Estatísticas descritivas
st.subheader('Estatísticas Descritivas')
st.write(df.describe())

st.markdown(''' ### Interpretação:

1. count: Todas as colunas têm 1000 registros. Isso indica que não há valores ausentes nos dados.

2. mean (média):
O preço médio dos produtos (price) é 247,67.
A pontuação média das avaliações (review_score) é 3,03 o que sugere que, em média, os produtos têm uma avaliação satisfatória.
As vendas médias por mês variam entre 487,19 e 514,80, mostrando uma certa consistência nas vendas ao longo dos meses.


3. std (desvio padrão):
O desvio padrão do preço (price) é 144,61, indicando que há uma grande variação nos preços dos produtos. Esse desvio representa 58% do valor medio do produto gerando uma grande dispeção nos preços.

4. A pontuação de avaliação tem um desvio padrão de 1,17, sugerindo alguma dispersão nas avaliações dos produtos.
As vendas mensais têm desvios padrão em torno de 280-290, o que pode indicar que há variações consideráveis nas vendas mensais entre os produtos.
''')


# Feature Engineering
df['mean_sales'] = df[[f'sales_month_{i}' for i in range(1, 13)]].mean(axis=1)
df['variance_sales'] = df[[f'sales_month_{i}' for i in range(1, 13)]].var(axis=1)
df['price_to_category_mean'] = df['price'] / df.groupby('category')['price'].transform('mean')
df['quantity_total_sold'] = df[[f'sales_month_{i}' for i in range(1, 13)]].sum(axis=1)
df['amount_total_sold'] = df['quantity_total_sold'] * df['price']
df['log_price'] = np.log1p(df['price'])
df['log_total_sales'] = np.log1p(df['amount_total_sold'])
df['price_range'] = pd.cut(df['price'], bins=[0, 100, 300, np.inf], labels=['Low', 'Medium', 'High'])
df['peak_month'] = df[[f'sales_month_{i}' for i in range(1, 13)]].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
df['price_review_interaction'] = df['price'] * df['review_score']

# Visualização 1: Boxplot de vendas médias por categoria
st.subheader('Vendas Médias por Categoria')
fig1 = px.box(df, x='category', y='mean_sales', title='Boxplot analise de vendas')
fig1.update_xaxes(tickangle=45)  # Rotaciona os rótulos do eixo x
st.plotly_chart(fig1)

st.markdown('''##### Conclusão boxplot
As categorias apresentam distribuições de vendas relativamente equilibradas, exceto Books, que mostra maior variabilidade e Electronics, que tem outliers indicando vendas pontuais altas.
Pode ser interessante investigar mais a fundo os outliers em Electronics para entender o que impulsionou as vendas nesses casos.
''')

# Visualização 2: Contagem de Produtos por Faixa de Preço
st.subheader('Contagem de Produtos por Faixa de Preço')
fig2 = px.histogram(df, x='price_range', title='Distribuição faixa de preço')
fig2.update_layout(xaxis_title='Faixa de Preço', yaxis_title='Contagem')
st.plotly_chart(fig2)

st.markdown('''##### Interpretação dos Padrões:

+ Distribuição da faixa de preço: Este grafico apenas confirma a analise de distribuição de valores, 
porem sem a plicação logarítmica nos preços, o que mostra consistência na conclusão da analise do conjuntos de graficos no 
'profile.html' onde conseguimos ver a adistribuição no log_total_sales e log_price. A distribuição do volume de produtos e valor estão 
classificados entre valores medios e altos em relação a todo volume de produtos, ou seja possuimos uma grande quantidade de produtos com preco medio e alta em relação a todos os produtos do e-comerce.

''')

# Analise de sazonalidade'
st.subheader('Mês de Pico de Vendas por Produto')
fig = px.histogram(df, x='peak_month', nbins=12, title='Analise sazonalidade')# Criando o gráfico interativo com Plotly
fig.update_layout(
    xaxis_title='Mês',
    yaxis_title='Contagem de Produtos',
    bargap=0.2
) #Adicionando rótulos e layout

# Exibindo o gráfico no Streamlit
st.plotly_chart(fig)

st.markdown('''##### Analise sazonalidade

Interpretação:

+ Mês 7 (Julho): É claramente o mês com o maior número de produtos atingindo seu pico de vendas. Isso pode indicar uma forte demanda sazonal, possivelmente devido a promoções de meio de ano ou férias.

+ Demais Meses: Há uma distribuição relativamente uniforme para os outros meses, com exceção do mês 12 (Dezembro), que tem a menor contagem de produtos atingindo o pico de vendas. Isso pode ser um indicativo de que alguns produtos não têm um aumento significativo de vendas durante o período natalino.
''')


# Visualização 4: Mapa de Calor - Categorias por Mês de Pico de Vendas
# Criando a tabela cruzada
category_peak_month = pd.crosstab(df['peak_month'], df['category'])

# Criando a figura do heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=category_peak_month.values,
    x=category_peak_month.columns,
    y=category_peak_month.index,
    colorscale='Blues',
    colorbar=dict(title='Contagem'),
    text=category_peak_month.values,
    texttemplate='%{text}',  # Formato para mostrar os números
    textfont=dict(size=12, color='white')
))

fig_heatmap.update_layout(
    title='Categorias de Produtos por Mês de Pico de Vendas',
    xaxis_title='Categoria',
    yaxis_title='Mês de Pico'
)

st.plotly_chart(fig_heatmap)

st.markdown('''##### Interpretação dos Padrões:

+ Com o mapa de calor conseguimos entender quais foram as categorias mais vendidades durantes os meses, é notavel a quantidade de vendas de eletronicos no mes 7 (Julho) o que impactou significativamente nas vendas do mês, devido a alta correlação. O mesmo impcto em eletetronicos ocorreu em Dezembro, porem o impacto foi o contrario do mês de Julho.
''')




# Visualização 5: Interação entre Preço e Avaliação
# Criando o scatter plot interativo
fig_scatter = px.scatter(df, x='price', y='price_review_interaction',
                        color='review_score', color_continuous_scale='viridis',
                        title='Interação entre Preço e Avaliação',
                        labels={'price': 'Preço', 'price_review_interaction': 'Interação Preço x Avaliação'})

fig_scatter.update_layout(coloraxis_colorbar=dict(title='Pontuação de Avaliação'))
st.plotly_chart(fig_scatter)

st.markdown('''##### Interpretação dos Padrões:

O gráfico mostra a relação entre o preço e a avaliação de um produto. O eixo x representa o preço do produto, e o eixo y representa a avaliação média do produto. A linha no gráfico mostra a tendência geral da relação entre o preço e a avaliação.

No geral, o gráfico mostra que há uma correlação positiva entre o preço e a avaliação do produto. Isso significa que, em geral, produtos com preços mais altos tendem a ter avaliações mais altas.


''')




# modelagem de dados:

# Aplicando One-Hot Encoding para as colunas 'category' e 'price_range'
df = pd.get_dummies(df, columns=['category', 'price_range'], prefix=['cat', 'price_range'])

# Definindo as variáveis independentes (X) e a variável dependente (y)
X = df.drop(columns=['amount_total_sold', 'product_name'])  # Excluindo colunas irrelevantes
y = df['amount_total_sold']  # Variável que desejamos prever

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Instanciando o modelo
rf_model = RandomForestRegressor(random_state=42)

# Treinando o modelo
rf_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_rf = rf_model.predict(X_test)

# Avaliação do modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf= mean_squared_error(y_test, y_pred_rf)
print(f'MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}')

# Instanciando o modelo
xgb_model = XGBRegressor(random_state=42)

# Treinando o modelo
xgb_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_xgb = xgb_model.predict(X_test)

# Avaliação do modelo
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost Regressor - MAE: {mae_xgb:.2f}, MSE: {mse_xgb:.2f}')


# Instanciando o modelo
linear_model = LinearRegression()

# Treinando o modelo
linear_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_lr = linear_model.predict(X_test)

# Avaliação do modelo
mae_linear = mean_absolute_error(y_test, y_pred_lr)
mse_linear = mean_squared_error(y_test, y_pred_lr)
print(f'Regressão Linear - MAE: {mae_linear: .2f}, MSE: {mse_linear: .2f}')


# Instanciando o modelo
gbr_model = GradientBoostingRegressor(random_state=42)

# Treinando o modelo
gbr_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_gbr = gbr_model.predict(X_test)

# Avaliação do modelo
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print(f'Gradient Boosting Regressor - MAE: {mae_gbr:.2f}, MSE: {mse_gbr:.2f}')

#Comparação de MAE e MSE entre Modelos


# Definindo os valores de MAE e MSE para cada modelo
model_names = ['Random Forest', 'Linear Regression', 'XGBoost Regressor', 'Gradient Boosting']
mae_values = [mae_rf, mae_linear, mae_xgb, mae_gbr]
mse_values = [mse_rf, mse_linear, mse_xgb, mse_gbr]

# Criando DataFrame para as métricas
metrics_df = pd.DataFrame({
    'Model': model_names,
    'MAE': mae_values,
    'MSE': mse_values
})

# Gráfico de MAE
fig_mae = go.Figure()
fig_mae.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'],
                         name='MAE',
                         marker_color='royalblue'))

fig_mae.update_layout(title='Comparação de MAE entre Modelos',
                      xaxis_title='Modelos',
                      yaxis_title='MAE')

# Gráfico de MSE
fig_mse = go.Figure()
fig_mse.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MSE'],
                         name='MSE',
                         marker_color='royalblue'))

fig_mse.update_layout(title='Comparação de MSE entre Modelos',
                      xaxis_title='Modelos',
                      yaxis_title='MSE')

# Exibindo os gráficos no Streamlit
st.plotly_chart(fig_mae)
st.plotly_chart(fig_mse)

st.markdown('''##### Conclusão do melhor modelo:

##### Interpretação dos Resultados
+ Random Forest: Apresentou os menores valores de MAE e MSE, indicando que é o modelo mais preciso entre os testados.
+ Regressão Linear: Teve os maiores valores de MAE e MSE, sugerindo que é o modelo menos preciso.
+ XGBoost Regressor e Gradient Boosting Regressor: Apresentaram resultados intermediários, com o Gradient Boosting sendo mais preciso que o XGBoost.


##### Análise dos Resultados

> MAE (Mean Absolute Error):

+ O MAE da Regressão Linear (99.960,98) é muito maior do que o da Random Forest (3.738,32). Isso indica que a Random Forest tem um desempenho significativamente melhor em termos de erro absoluto médio.


>MSE (Mean Squared Error):

+ O MSE da Regressão Linear (19.166.618.609,47) também é muito superior ao da Random Forest (40.743.711,66), mostrando que a Regressão Linear possui um erro quadrático médio bem maior, o que sugere que ela tem mais dificuldades em prever com precisão os valores extremos.


''')





# Comparação das Vendas Reais e Previstas

# Lista de previsões de cada modelo
model_predictions = {
    'Random Forest': y_pred_rf,
    'Regressão Linear': y_pred_lr,
    'XGBoost Regressor': y_pred_xgb,
    'Gradient Boosting Regressor': y_pred_gbr
}

# Criando a figura e os eixos para os 4 gráficos (2x2 layout)
fig_comp = sp.make_subplots(rows=2, cols=2, subplot_titles=list(model_predictions.keys()))

for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
    row = idx // 2 + 1
    col = idx % 2 + 1
    fig_comp.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Previsões'),
                       row=row, col=col)
    fig_comp.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines',
                                  name='Linha de Referência', line=dict(color='black', dash='dash')),
                       row=row, col=col)
    fig_comp.update_xaxes(title_text='Vendas Reais', row=row, col=col)
    fig_comp.update_yaxes(title_text='Vendas Previstas', row=row, col=col)
    fig_comp.update_layout(title_text=f'Comparação das Vendas Reais e Previstas - {model_name}')

fig_comp.update_layout(height=800, width=800, title_text='Comparação das Vendas Reais e Previstas por Modelo')

# Exibindo os gráficos no Streamlit
st.plotly_chart(fig_comp)


##  Execute o seguinte codigo no terminal para abrir o app: streamlit run app.py
