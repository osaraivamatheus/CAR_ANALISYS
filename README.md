# Carregando pacotes e configurando funções


```python
#Carregando pacotes
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf #pip install yfinance
import numpy as np
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #grafico acf pacf
from statsmodels.stats.diagnostic import acorr_ljungbox #teste de independencia residual
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro, normaltest, probplot

import warnings #apenas para remover avisos
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
#warnings.simplefilter(action='ignore', category=FutureWarning)
    
##para datas em portugues
import matplotlib.pyplot as plt
import locale
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8") 
plt.rcParams['axes.formatter.use_locale'] = True
from matplotlib.dates import DateFormatter
formato = DateFormatter('%b, %Y')

##para fontes mais bonitas nos graficos
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 250
plt.rcParams['axes.formatter.use_locale'] = True # ',' como decimal nos graficos

#### graficos bonitinhos
def eixo_seta():
    ax = plt.gca()
    fig = plt.gcf()
    # removing the default axis on all sides:
    for side in ['right','top']:
        ax.spines[side].set_visible(False)

    plt.xticks(fontsize=16,rotation=0, ha='center')
    plt.yticks(fontsize=16)
    plt.xlabel('')

#Funcao que calcula retornos anormais acumulados
def r(dados):
    #RETORNO IBOVESPA
    ibov = np.diff(dados.iloc[:,1])
    leg_ibov = dados.iloc[:,1][:-1]
    ret_ibov = ibov / leg_ibov - 1
    
    #RETORNO SA
    sa = np.diff(dados.iloc[:,0])
    leg_sa = dados.iloc[:,0][:-1]
    ret_sa = sa / leg_sa - 1
    
    return np.cumsum(ret_sa - ret_ibov)

#Coletando dados e formando banco de dados
def m_dados(bancos, i, f, freq='1d'):
    bancos.append('^BVSP')
    #Coletando dados diarios de fechamento
    dados = yf.download(bancos, start=i, end=f, progress=False,interval=freq)['Close'].dropna().reset_index()

    #Montando data frame    
    if len(bancos)>1:
        sa = r(dados.loc[:, [bancos[0], '^BVSP']])
        sa = pd.DataFrame({bancos[0]:sa})
        for i in range(1, (len(bancos)-1)):
            sa1 = r(dados.loc[:, [bancos[i],'^BVSP']])
            sa = pd.concat([sa, sa1], axis = 1)
       
    else:
        sa = r(dados.loc[:, [bancos[0],'^BVSP']])
        sa = pd.DataFrame({bancos[0]:sa})
        
        
    sa.set_index(dados['Date'][1:], inplace = True)
    bancos.pop()
    sa.columns = bancos
    
    
    return sa
```

# Coletando dados e plotando gráfico de retornos


```python
#Coletando dados e plotando
empresas = ['BBAS3.SA', 'BBDC4.SA', 'ITSA4.SA'] #definindo bancos
inicio = '2008-01-03'
fim = '2021-01-08'
covid = datetime.date(2020,2,26)
dados = m_dados(empresas, inicio, fim, freq='1d')
cores = ['darkblue', 'darkred', 'darkorange']
dados.plot(figsize = [16,10], color = cores);
plt.axvline(x=covid, ls='--', color='black')
plt.annotate('Início Covid-19', (datetime.date(2020,3,26), .8))
leg = plt.legend(fontsize=20)
for line in leg.get_lines():
    line.set_linewidth(8.0)
plt.ylabel('CAR', fontsize=18);
eixo_seta()

```



# Coletando dados Ibovespa


```python
bovespa = yf.download(['^BVSP'] ,period='max', progress=False)['Close'].dropna()
bovespa.plot(figsize = [16,10], label='Bovespa', color='black');
plt.ylabel('Índice', fontsize=18)
plt.xticks(rotation=0, ha = 'center');

bovespa = pd.DataFrame(bovespa)
bovespa['ano'] = pd.DatetimeIndex(bovespa.index).year

menores_x = []
for i in [2008,2020]:
    x = bovespa.groupby('ano').min().loc[i, 'Close']
    filt = (bovespa['Close'] == x)
    x = bovespa[filt].index
    menores_x.append(x)
    
y1 = bovespa.groupby('ano').min().loc[2008, 'Close']
y2 = bovespa.groupby('ano').min().loc[2020, 'Close']
ys = [y1,y2]

plt.scatter(menores_x, ys, color='red', marker='o')
plt.annotate('Crise financeira \n de 2008', (menores_x[0], ys[0]-7000))
plt.annotate('Pandemia \n Covid-19', (menores_x[1], ys[1]-7000))

#### colocando separador de milhar no eixo y:
ax = plt.gca()
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, loc: locale.format('%d', x, 1)))
eixo_seta()

```


# Calculando estatísticas descritivas


```python
# Criando tabela de estatisticas descritivas
medias = np.round(dados.apply(np.mean, axis=0), 4)
variancias = np.round(dados.apply(np.var, axis=0), 4)
erro_padrao = np.round(dados.apply(np.std, axis=0), 4)
minimos = np.round(dados.apply(np.min, axis=0),4)
maximos = np.round(dados.apply(np.max, axis=0),4)
bovespa = bovespa['Close']
bovespa = np.diff(bovespa) / bovespa[:-1] - 1 

media_bov = np.round(np.mean(bovespa), 4)
var_bov = np.round(np.var(bovespa),4)
std_bov = np.round(np.var(bovespa),4)
min_bov = np.round(np.min(bovespa),4)
max_bov = np.round(np.max(bovespa),4)


tabela = pd.concat([minimos, maximos, medias, variancias, erro_padrao], axis=1)
tabela.loc['Bovespa'] = [min_bov, max_bov, media_bov, var_bov, std_bov]
tabela.columns=['Mínimo','Máximo','Média','Variância','Desvio Padrão']
#print(tabela.to_latex())
#print(tabela.to_markdown())
tabela
```

# Boxplot dos retornos


```python
dados.boxplot(figsize=[16,5], grid=False);
```

# Autocorrelação dos retornos de cada série


```python
fig, ax = plt.subplots(1, 3, figsize = (16,5))
for i in range(3):
    plot_acf(dados[empresas[i]],  
             ax= ax[i], 
             lags=50,
             title='Autocorrelação da série \n '+ empresas[i])
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

```

# FAC dos retornos das séries diferenciadas


```python
fig, ax = plt.subplots(1, 3, figsize = (16,5))
for i in range(3):
    plot_acf(np.diff(dados[empresas[i]], 1), zero=False,
              ax= ax[i], 
              title='Autocorrelação da série '+ empresas[i]+ ' \n diferenciada')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
```

# Teste de raiz unitária (Dickey Fuller aumentado) para as séries diferenciadas


```python
dados_dif = dados.apply(np.diff, axis = 0)

adf_empresa1 = adfuller(dados_dif[empresas[0]], )
adf_empresa2 = adfuller(dados_dif[empresas[1]])
adf_empresa3 = adfuller(dados_dif[empresas[2]])

serie = empresas
ADF = [adf_empresa1[0], adf_empresa2[0], adf_empresa2[0]]
vp = [adf_empresa1[1], adf_empresa2[1], adf_empresa2[1]]

x = pd.DataFrame({'Est. ADF': ADF, 'Valor-p':vp}, index=serie)
#print(x.to_markdown())
```

# Gráficos das séries diferenciadas


```python
fig, ax = plt.subplots(3,1, figsize=[16,12])
for i in range(3):
    ax[i].plot(dados[empresas[i]].diff(), label=None, color = '#727272');
    eixo_seta()
    ax[i].set_title(empresas[i]);
    ax[i].axhline(y=0, ls='--', color='black')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].tick_params(axis="x", labelsize=16) 
    ax[i].tick_params(axis="y", labelsize=16)
    ax[i].set_xlabel('')
```

# FAC e FACP das séries diferenciadas


```python
fig, ax = plt.subplots(2, 3, figsize = (16,12))
for j in range(3):
    plot_acf(np.diff(dados[empresas[j]]),
              zero=False,
              lags=50,
              ax= ax[0,j], 
              title='Autocorrelação da série '+ empresas[j] + '\n diferenciada')
    plot_pacf(np.diff(dados[empresas[j]], 1),
              zero=False,
              lags=50,
              ax= ax[1,j], 
              title='Autocorrelação parcial da série '+ empresas[j] + '\n diferenciada')
    ax[0,j].spines['top'].set_visible(False)
    ax[0,j].spines['right'].set_visible(False)
    ax[1,j].spines['top'].set_visible(False)
    ax[1,j].spines['right'].set_visible(False)

```
# AIC e EQM dos primeiros modelos


```python
def eqm(x,y):
    return sum((x-y)**2)/len(x)

aicsm1 = []
eqmsm1 = []

aicsm2 = []
eqmsm2 = []

ordensM1 = [(1,0,0), (1,0,1), (1,0,1)]
ordensM2 = [([2,3],1,[2]), (0,1,[1,2,3,6]), (1,1,[1,6])]

for i in range(3):
    m1 = ARIMA(dados.iloc[:, i], order = ordensM1[i]).fit()
    errom1 = m1.resid
    aicsm1.append(m1.aic)
    eqmsm1.append(eqm(dados.iloc[:,i], errom1))
    
    m2 = ARIMA(dados.iloc[:, i], order = ordensM2[i]).fit()
    errom2 = m2.resid
    aicsm2.append(m2.aic)
    eqmsm2.append(eqm(dados.iloc[:,i], errom2))
    
pd.DataFrame({'aic m1':aicsm1, 
              'aic m2':aicsm2,
              'eqm m1': eqmsm1,
               'eqm m2': eqmsm2})
```

# Resultados dos novos modelos


```python
vetor_datas = m_dados(empresas, inicio, fim, freq='1d').index
dados.index = vetor_datas
dados.index =pd.DatetimeIndex(dados.index).to_period('D')

####ordens ajustadas:
# BRASIL = ARIMA(3,2,2) com defasagens especificas nos lags 2 e 3
# BRADESCO = ARIMA(0,1,6) com desfasagens especificas nos lags 1,2,3,6
# ITAU = ARIMA(1,1,6) com defasagens especificas nos lags 1, 6

resumos = []

###funcao que calcula o eqm

ordens = [([2,3],1,[2]), (0,1,[1,2,3,6]), (1,1,[1,6])]
fig, ax = plt.subplots(3,1, figsize=[16,12])

quant = 22 ##quantidade de dados para treino
for i in range(3):
    treino = dados.iloc[:-quant, i]
    teste = dados.iloc[-quant:, i]
    
    modelo = ARIMA(treino, order=ordens[i]).fit()
    r = modelo.summary()
    r = r.tables[1].as_html()
    r = pd.read_html(r, header=0, index_col=0)[0]
    resumos.append(r)
    
    f = modelo.get_forecast(quant)
    previsoes = f.predicted_mean
    int_conf = f.conf_int(alpha = .05)
    f = pd.concat([previsoes, int_conf], axis=1)
    f.columns = ['pred','lower','upper']
    
    treino.index = vetor_datas[:-quant]
    teste.index = vetor_datas[-quant:]
    f.index = vetor_datas[-quant:]
    

    # Plot    
    ax[i].plot(treino[-50:], label='Treino')
    ax[i].plot(teste, label='Teste')
    ax[i].plot(f['pred'], label='Previsão')
    ax[i].fill_between(f.index, f['lower'], f['upper'], label='IC = 95\%', color='gray', alpha=.15)
    ax[i].set_title(empresas[i])
    ax[i].legend(loc='upper left', fontsize=8)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

```

# Análise de resíduos dos modelos ajustados


```python
fig, ax = plt.subplots(3,1, figsize=[16,17],dpi=250)

resid_test = pd.DataFrame()
for i in range(3):
    modelo = ARIMA(dados[empresas[i]], order=ordens[i]).fit()

    resid_test[empresas[i]] = modelo.resid
    
    # Plot    
    plot_acf(modelo.resid, 
             ax = ax[i], 
             zero = False, 
             title = 'Autocorrelação dos resíduos do modelo ARIMA' + str(ordens[i]) + '\n ajustados para a série ' + empresas[i])
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
```

# Gráfico dos ajustes


```python
dados.index = vetor_datas
dados.index =pd.DatetimeIndex(dados.index).to_period('D')

####ordens ajustadas:
# BRASIL = ARIMA(3,2,2) com defasagens especificas nos lags 2 e 3
# BRADESCO = ARIMA(0,1,6) com desfasagens especificas nos lags 1,2,3,6
# ITAU = ARIMA(1,1,6) com defasagens especificas nos lags 1, 6

ordens = [([2,3],1,[2]), (0,1,[1,2,3,6]), (1,1,[1,6])]

fig, ax = plt.subplots(1,1, figsize=[16,5])

res = pd.DataFrame(columns = [empresas[0], empresas[1], empresas[2]])
for i in range(3):
    dados.index =pd.DatetimeIndex(vetor_datas).to_period('D')
    modelo = ARIMA(dados[empresas[i]], order=ordens[i]).fit(method='statespace')
    #guardando residuos
   
    res[empresas[i]] = modelo.resid
    # Plot
    dados.index = vetor_datas
    ax.plot(dados[empresas[i]][-50:], label=empresas[i], color=cores[i])
    ax.plot(modelo.fittedvalues[-50:], ls='--', color = cores[i], label='ajustado')
    ax.legend(loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

```
# Teste de independência dos resíduos (Ljung-Box)


```python
plt.subplots(1,1, figsize=[16,5])
for i in range(3):
    lb = acorr_ljungbox(res[empresas[i]], return_df=True, )['lb_pvalue']
    plt.plot(lb, label=empresas[i], color=cores[i])
    plt.ylabel('Valor-p')
    plt.xlabel(r'$lag$')
plt.axhline(y=.05, ls='--', label = r'$\alpha = 5\%$')
plt.legend(loc='upper right')
eixo_seta()
```

# Histograma dos retornos de cada série


```python
fig, ax = plt.subplots(1,3, figsize=[16,5])
for i in range(3):
    ax[i].hist(dados[empresas[i]], color='white', edgecolor='C0');
    ax[i].set_title(empresas[i]);
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

```
