import streamlit as st
import pandas as pd
import re
import inspect
import datetime
import time
import plotly.express as px
import numpy as np

# --- Tratamento de Dados ---
file_path = r"/workspaces/Streamlit-DA/Global.Dataset.DAEconomics.Hard.Data.xlsm"
df = pd.read_excel(file_path, sheet_name="HARD.DATA", engine="openpyxl")

# Remove colunas 'Unnamed'
df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
cols_unnamed = [c for c in df.columns if 'Unnamed' in c]
df = df.drop(columns=cols_unnamed)

# Remove as linhas de cabeçalho extras e reindexa
df = df.drop(index=[0, 1, 2, 4]).reset_index(drop=True)

# --- Geração Dinâmica de Funções de Série ---
series_vars = []
cols = df.columns.tolist()
for i, col in enumerate(cols):
    if col.startswith("Security") and i + 1 < len(cols):
        val_col = cols[i + 1]
        series_name = df.at[0, col]
        if pd.notna(series_name):
            fn_name = re.sub(r'\W|^(?=\d)', '_', str(series_name))
            def make_series_fn(sec_col, val_col, clean_name):
                def series_fn():
                    tmp = df[[sec_col, val_col]].iloc[1:].copy().reset_index(drop=True)
                    tmp.columns = ["Date", clean_name]
                    return tmp.dropna(subset=[clean_name])
                return series_fn
            globals()[fn_name] = make_series_fn(col, val_col, fn_name)
            series_vars.append(series_name)

# --- Dashboard Streamlit ---
st.set_page_config(page_title="DA Economics – Dashboard de Séries", layout="wide")
st.title("📊 Dashboard de Séries DA Economics")

# Coleta funções geradas
def is_series_fn(obj):
    return callable(obj) and not obj.__name__.startswith("_") and len(inspect.signature(obj).parameters)==0
var_funcs = {name: func for name, func in globals().items() if is_series_fn(func)}

# 1. Seleção da série
grupos = ["Equity Market", "FX Market", "Commodities Market", "Bonds Market"]
serie = st.sidebar.selectbox("Selecione a Série:", sorted(var_funcs.keys()))
categoria = st.sidebar.selectbox("Selecione o grupo da série:", grupos)

# 2. Prepara dados da série
fn = var_funcs[serie]
df_sel = fn()
valor_col = df_sel.columns[1]
df_sel[valor_col] = pd.to_numeric(df_sel[valor_col], errors='coerce')
df_sel['Date'] = pd.to_datetime(df_sel['Date'])

# 3. Agora sim, use o slider com as datas corretas
st.sidebar.markdown("**Selecione o Período:**")
min_date = df_sel['Date'].min().date()
max_date = df_sel['Date'].max().date()
date_range = st.sidebar.slider(
    "Período",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="DD/MM/YYYY"
)

# Indicadores no gráfico principal
st.sidebar.markdown("**Indicadores no Gráfico Principal:**")
show_ma_3m = st.sidebar.checkbox("Média Móvel 3M", value=True)
show_ma_6m = st.sidebar.checkbox("Média Móvel 6M")
show_ma_12m = st.sidebar.checkbox("Média Móvel 12M")
show_var_diario = st.sidebar.checkbox("Var Diário")
show_var_mensal = st.sidebar.checkbox("Var Mensal")
show_var_anual = st.sidebar.checkbox("Var Anual")
show_acc_3m = st.sidebar.checkbox("Acc 3M")
show_acc_6m = st.sidebar.checkbox("Acc 6M")
show_acc_12m = st.sidebar.checkbox("Acc 12M")
show_indice_original = st.sidebar.checkbox("Índice Original", value=True)

# Gráficos separados
st.sidebar.markdown("---")
st.sidebar.markdown("**Gráficos Separados:**")
show_macd = st.sidebar.checkbox("Mostrar Gráfico MACD")
show_rsi = st.sidebar.checkbox("Mostrar Gráfico RSI")
show_volume = st.sidebar.checkbox("Mostrar Gráfico de Volume Separado")
show_slider = st.sidebar.checkbox("Range Slider nos Gráficos", value=True)

# Seleção de métricas
options = ["Índice Original", "Média Móvel 3M", "Média Móvel 6M", "Média Móvel 12M",
           "Var Diário", "Var Mensal", "Var Anual",
           "Acc 3M", "Acc 6M", "Acc 12M"]

# Mapeamento das opções para as variáveis de checkbox
checkbox_map = {
    "Índice Original": show_indice_original,
    "Média Móvel 3M": show_ma_3m,
    "Média Móvel 6M": show_ma_6m,
    "Média Móvel 12M": show_ma_12m,
    "Var Diário": show_var_diario,
    "Var Mensal": show_var_mensal,
    "Var Anual": show_var_anual,
    "Acc 3M": show_acc_3m,
    "Acc 6M": show_acc_6m,
    "Acc 12M": show_acc_12m,
}

selected = [m for m in options if checkbox_map[m]]

# Define eixo para cada métrica
axis_assign = {}
for metric in selected:
    axis_assign[metric] = st.sidebar.radio(f"Eixo para {metric}:", ["Primário", "Secundário"], index=0)

# 1. Filtra e ordena pelo período selecionado
start_date, end_date = date_range
df_plot = df_sel.loc[
    (df_sel['Date'] >= pd.to_datetime(start_date)) &
    (df_sel['Date'] <= pd.to_datetime(end_date))
].sort_values('Date').reset_index(drop=True)

# 2. Prepara DataFrame de plotagem e tabela
plot_df = pd.DataFrame({"Date": df_plot['Date'], serie: df_plot[valor_col]})

# 3. Cálculos (agora sobre df_plot, que está ordenado corretamente)
metrics = [m for m in selected if m != "Índice Original"]
for m in metrics:
    if m == "Média Móvel 3M": plot_df['MA_3M'] = df_plot[valor_col].rolling(63).mean().values
    if m == "Média Móvel 6M": plot_df['MA_6M'] = df_plot[valor_col].rolling(126).mean().values
    if m == "Média Móvel 12M": plot_df['MA_12M'] = df_plot[valor_col].rolling(252).mean().values
    if m == "Var Diário": plot_df['Var_D'] = df_plot[valor_col].pct_change(1).values
    if m == "Var Mensal": plot_df['Var_M'] = df_plot[valor_col].pct_change(21).values
    if m == "Var Anual": plot_df['Var_A'] = df_plot[valor_col].pct_change(252).values
    if m == "Acc 3M": plot_df['Acc_3M'] = (df_plot[valor_col]/df_plot[valor_col].shift(3)-1).values
    if m == "Acc 6M": plot_df['Acc_6M'] = (df_plot[valor_col]/df_plot[valor_col].shift(6)-1).values
    if m == "Acc 12M": plot_df['Acc_12M'] = (df_plot[valor_col]/df_plot[valor_col].shift(12)-1).values

# Ordena por data ANTES de exibir ou plotar
plot_df = plot_df.sort_values('Date').reset_index(drop=True)

# KPIs SEMPRE calculados, independente dos checkboxes
indice_atual = df_plot[valor_col].iloc[-1] if not df_plot.empty else None

# Var. Diária
if categoria == "Bonds Market":
    # Cálculos em basis points (bps)
    if len(df_plot) > 1:
        var_diaria = (df_plot[valor_col].iloc[-1] - df_plot[valor_col].iloc[-2]) * 100
    else:
        var_diaria = None
else:
    # Cálculos padrão (percentual)
    if len(df_plot) > 1:
        var_diaria = (df_plot[valor_col].iloc[-1] - df_plot[valor_col].iloc[-2]) / df_plot[valor_col].iloc[-2]
    else:
        var_diaria = None

# Var. Mensal (21 dias úteis atrás)
if len(df_plot) > 21:
    if categoria == "Bonds Market":
        var_mensal = (df_plot[valor_col].iloc[-1] - df_plot[valor_col].iloc[-22]) * 100
    else:
        var_mensal = (df_plot[valor_col].iloc[-1] - df_plot[valor_col].iloc[-22]) / df_plot[valor_col].iloc[-22]
else:
    var_mensal = None

# Var. Anual (252 dias úteis atrás)
if len(df_plot) > 252:
    if categoria == "Bonds Market":
        var_anual = (df_plot[valor_col].iloc[-1] - df_plot[valor_col].iloc[-253]) * 100
    else:
        var_anual = (df_plot[valor_col].iloc[-1] - df_plot[valor_col].iloc[-253]) / df_plot[valor_col].iloc[-253]
else:
    var_anual = None

# YTD
if not df_plot.empty:
    today = df_plot['Date'].max()
    year_start = pd.Timestamp(year=today.year, month=1, day=1)
    df_year_start = df_plot[df_plot['Date'] >= year_start]
    if not df_year_start.empty:
        idx_year_start = df_year_start.index[0]
        valor_year_start = df_plot.iloc[idx_year_start][valor_col]
        valor_hoje = df_plot.iloc[-1][valor_col]
        if categoria == "Bonds Market":
            ytd = (valor_hoje - valor_year_start) * 100 if valor_year_start != 0 else None
        else:
            ytd = (valor_hoje - valor_year_start) / valor_year_start if valor_year_start != 0 else None
    else:
        ytd = None
else:
    ytd = None

# Exibe os KPIs fixos
col1, col2, col3, col4, col5 = st.columns(5)

def colored_percent(val):
    if val is None:
        return "N/A"
    if categoria == "Bonds Market":
        color = "#00FF00" if val > 0 else "#FF0000" if val < 0 else "#FFFFFF"
        return f"<span style='color:{color}; font-size:2rem; font-weight:bold'>{val:,.2f} bps</span>"
    else:
        color = "#00FF00" if val > 0 else "#FF0000" if val < 0 else "#FFFFFF"
        return f"<span style='color:{color}; font-size:2rem; font-weight:bold'>{val:.2%}</span>"

def big_index(val):
    if val is None:
        return "N/A"
    return f"<span style='color:#FFFFFF; font-size:2.5rem; font-weight:bold'>{val:,.2f}</span>"

col1.markdown(f"<div style='text-align:center'><b>Índice Atual</b><br>{big_index(indice_atual)}</div>", unsafe_allow_html=True)
col2.markdown(f"<div style='text-align:center'><b>Var. Diária</b><br>{colored_percent(var_diaria)}</div>", unsafe_allow_html=True)
col3.markdown(f"<div style='text-align:center'><b>Var. Mensal</b><br>{colored_percent(var_mensal)}</div>", unsafe_allow_html=True)
col4.markdown(f"<div style='text-align:center'><b>Var. Anual</b><br>{colored_percent(var_anual)}</div>", unsafe_allow_html=True)
col5.markdown(f"<div style='text-align:center'><b>YTD</b><br>{colored_percent(ytd)}</div>", unsafe_allow_html=True)

# Gráfico interativo com Plotly
import plotly.graph_objects as go
fig = go.Figure()
for col in plot_df.columns:
    if col == 'Date': continue
    yaxis = 'y2' if any(col in k for k,v in axis_assign.items() if v=='Secundário' and (col.startswith(k.split()[0]) or col==k or col in ['Var_D','Var_M','Var_A','Acc_3M','Acc_6M','Acc_12M'])) else 'y'
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df[col], name=col, yaxis=yaxis))
fig.update_layout(
    title=f"{serie} e métricas selecionadas",
    xaxis_title='Date', yaxis_title='Valor',
    yaxis2=dict(overlaying='y', side='right', title='Percentual'),
    dragmode='zoom', hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Exibe tabela sempre presente
st.subheader("📋 Tabela: Dados e Métricas Selecionadas")
st.dataframe(plot_df, use_container_width=True)

# 1. Pegue os últimos 20 anos de dados
df_heat = df_sel.copy()
df_heat['Date'] = pd.to_datetime(df_heat['Date'])
df_heat = df_heat.sort_values('Date')
df_heat['Year'] = df_heat['Date'].dt.year
df_heat['Month'] = df_heat['Date'].dt.month

# 2. Filtra para os últimos 20 anos
anos_disponiveis = sorted(df_heat['Year'].unique())
anos_heat = anos_disponiveis[-20:]
df_heat = df_heat[df_heat['Year'].isin(anos_heat)]

# 3. Pega o último valor de cada mês
df_monthly = df_heat.groupby(['Year', 'Month'])[valor_col].last().reset_index()

# 4. Calcula o retorno mensal
if categoria == "Bonds Market":
    df_monthly['Retorno'] = (df_monthly[valor_col] - df_monthly[valor_col].shift(1)) * 100
else:
    df_monthly['Retorno'] = df_monthly.groupby('Year')[valor_col].pct_change() * 100

# 5. Pivot para matriz (heatmap)
heatmap_data = df_monthly.pivot(index='Month', columns='Year', values='Retorno')
heatmap_data = heatmap_data.reindex(index=range(1,13))  # Garante ordem dos meses

# 6. Nomes dos meses
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
heatmap_data.index = meses

# 7. Cria matriz de textos coloridos
text_matrix = np.empty(heatmap_data.shape, dtype=object)
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.iloc[i, j]
        if pd.isna(val):
            text_matrix[i, j] = ""
        elif val < 0:
            text_matrix[i, j] = f"<span style='color:#FF3333; font-weight:bold'>{val:.2f}</span>"
        else:
            text_matrix[i, j] = f"<span style='color:#FFFFFF; font-weight:bold'>{val:.2f}</span>"

# 8. Plota o heatmap
heatmap_title = "Heatmap de Retornos Mensais por Ano"
if categoria == "Bonds Market":
    heatmap_title += " (em bps)"

fig_heat = px.imshow(
    heatmap_data,
    labels=dict(x="Ano", y="Mês", color="Retorno (%)"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale="Blues",
    aspect="auto",
    text_auto=False
)
fig_heat.update_traces(
    text=text_matrix,
    texttemplate="%{text}",
    textfont_size=18  # Aumenta o tamanho da fonte
)
fig_heat.update_layout(
    title="Heatmap de Retornos Mensais",
    width=1400,
    height=700
)

st.subheader("Heatmap de Retornos Mensais por Ano")
st.plotly_chart(fig_heat, use_container_width=True)



