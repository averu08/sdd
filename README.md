import pandas as pd
import numpy as np

# Dados fornecidos
dados = {
    'Mercado': [2.0, 1.5, -1.0, 3.0, 0.5, -2.0, 2.5, 1.0, -1.5, 3.5, 0.0, -2.5],
    'Acao A': [2.5, 2.0, -1.5, 3.5, 1.0, -3.0, 3.0, 1.5, -2.0, 4.0, 0.5, -3.5],
    'Acao B': [1.5, 1.0, -0.5, 2.0, 0.0, -1.5, 2.0, 0.5, -1.0, 2.5, -0.5, -2.0],
    'Acao C': [3.0, 2.5, -2.0, 4.0, 1.5, -3.5, 3.5, 2.0, -2.5, 4.5, 1.0, -4.0],
    'Acao D': [-1.0, 0.5, 1.0, 0.0, -0.5, 1.5, -0.5, 0.0, 1.0, -1.0, -0.5, 2.0]
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Função para calcular o Drawdown
def calculate_drawdown(returns):
    cumulative = (1 + returns / 100).cumprod()
    max_return = cumulative.cummax()
    drawdown = (cumulative - max_return) / max_return
    return drawdown.min() * 100, drawdown

# Função para calcular o Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0):
    excess_return = returns.mean() - risk_free_rate
    return excess_return / returns.std()

# Função para calcular o Sortino Ratio
def sortino_ratio(returns, risk_free_rate=0):
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std()
    excess_return = returns.mean() - risk_free_rate
    return excess_return / downside_std

# Função para calcular o Ulcer Index (UPI)
def ulcer_index(returns):
    drawdown = calculate_drawdown(returns)[1]
    ulcer_index = np.sqrt(np.mean(drawdown ** 2))
    return ulcer_index

# Calculando métricas para cada ação
acoes = ['Acao A', 'Acao B', 'Acao C', 'Acao D']

resultados = {}

for acao in acoes:
    retornos = df[acao]

    # Maximum Drawdown
    max_dd, _ = calculate_drawdown(retornos)

    # Sharpe Ratio
    sharpe = sharpe_ratio(retornos)

    # Sortino Ratio
    sortino = sortino_ratio(retornos)

    # Ulcer Index (UPI)
    ulcer = ulcer_index(retornos)

    # Guardar resultados
    resultados[acao] = {
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'UPI': ulcer,
        'Maximum Drawdown (%)': max_dd
    }

# Exibir resultados
pd.DataFrame(resultados).T


import matplotlib.pyplot as plt

# Dados dos resultados
resultados_df = pd.DataFrame({
    'Sharpe Ratio': [0.258, 0.225, 0.275, 0.211],
    'Sortino Ratio': [0.730, 0.511, 0.913, 0.761],
    'UPI': [0.015, 0.009, 0.018, 0.006],
    'Maximum Drawdown (%)': [-3.50, -2.49, -4.00, -1.50]
}, index=['Acao A', 'Acao B', 'Acao C', 'Acao D'])

# Configuração de subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Gráfico de Sharpe Ratio
resultados_df['Sharpe Ratio'].plot(kind='bar', ax=axes[0, 0], color='skyblue', title='Sharpe Ratio')
axes[0, 0].set_ylabel('Sharpe Ratio')

# Gráfico de Sortino Ratio
resultados_df['Sortino Ratio'].plot(kind='bar', ax=axes[0, 1], color='lightgreen', title='Sortino Ratio')
axes[0, 1].set_ylabel('Sortino Ratio')

# Gráfico de UPI (Ulcer Performance Index)
resultados_df['UPI'].plot(kind='bar', ax=axes[1, 0], color='salmon', title='Ulcer Index')
axes[1, 0].set_ylabel('Ulcer Performance Index')

# Gráfico de Maximum Drawdown
resultados_df['Maximum Drawdown (%)'].plot(kind='bar', ax=axes[1, 1], color='orange', title='Maximum Drawdown (%)')
axes[1, 1].set_ylabel('Maximum Drawdown (%)')

# Ajustar layout
plt.tight_layout()
plt.show()
