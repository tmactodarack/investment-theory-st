import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import math
plt.style.use('bmh')
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('retina')
import matplotlib as mpl
COLOR = 'dimgrey'
mpl.rcParams['text.color'] = COLOR



######################### Basic calculation function##########################
def port_ret_annual(weights, ret):
    port_ret_annual = np.sum(ret.mean()*weights) *252
    return port_ret_annual

def port_neg_ret_annual(weights, ret): # for finding maximum portfolio return
    return -port_ret_annual(weights, ret)

def port_std_annual(weights, cov_matrix):
    port_std_annual = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    return port_std_annual

# If we pass in ret and cov_matrix before rf adjusted, use this function
def port_negSharp(weights, ret, cov_matrix, rf):
    port_ret_annual = np.sum(ret.mean()*weights) *252
    port_std_annual = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    negSharp = - (port_ret_annual - rf) / port_std_annual
    return negSharp
######################### Basic calculation function##########################

# # If we pass in the excess ret and cov_max (prehandle the rf), use this instead
# def port_excess_ret_negSharp(weights, ret, cov_matrix):
#     port_ret_annual = ( np.sum(ret * weights) ).mean() * 252
#     port_std_annual = np.sqrt( weights.T @ cov_matrix @ weights ) * np.sqrt(252)
#     negSharp = - port_ret_annual / port_std_annual
#     return negSharp
######################### Some optimal portfolios ##########################
def max_ret_port(weights, ret, cov_matrix, rf):
    
    constraints = {'type':'eq', 'fun': lambda x: np.sum(x)-1}
    bounds = [(0,1)]*len(tickers)
    results = minimize(port_neg_ret_annual, weights, args=(ret), method='SLSQP', constraints=constraints, bounds=bounds)
    port_ret = port_ret_annual(results.x, ret)
    port_std = port_std_annual(results.x, cov_matrix)
    sharp = - port_negSharp(results.x, ret, cov_matrix, rf)

    weights_result = pd.Series(results.x, index=tickers)
    general_result = pd.Series([port_ret, port_std, sharp], index=['Return', 'Std', 'Sharp Ratio'])    

    single = pd.concat([weights_result, general_result]).to_frame('max ret')
    return single

def min_std_port(weights, ret, cov_matrix, rf):
    constraints = [{'type':'eq', 'fun': lambda x: np.sum(x)-1}, {'type':'ineq', 'fun':lambda x: port_ret_annual(x, ret)}]
    bounds = [(0,1)]*len(tickers)
    results = minimize(port_std_annual, weights, args=(cov_matrix), method='SLSQP', constraints=constraints, bounds=bounds)
    port_ret = port_ret_annual(results.x, ret)
    port_std = port_std_annual(results.x, cov_matrix)
    sharp = - port_negSharp(results.x, ret, cov_matrix, rf)

    weights_result = pd.Series(results.x, index=tickers)
    general_result = pd.Series([port_ret, port_std, sharp], index=['Return', 'Std', 'Sharp Ratio'])    

    single = pd.concat([weights_result, general_result]).to_frame('min std')
    return single

def max_sharp_port(weights, ret, cov_matrix, rf):
    constraints = [{'type':'eq', 'fun': lambda x: np.sum(x)-1}, {'type':'ineq', 'fun':lambda x: port_ret_annual(x, ret)}]
    bounds = [(0,1)]*len(tickers)
    results = minimize(port_negSharp, weights, args=(ret, cov_matrix, rf), method='SLSQP', constraints=constraints, bounds=bounds)
    port_ret = port_ret_annual(results.x, ret)
    port_std = port_std_annual(results.x, cov_matrix)
    sharp = - port_negSharp(results.x, ret, cov_matrix, rf)

    weights_result = pd.Series(results.x, index=tickers)
    general_result = pd.Series([port_ret, port_std, sharp], index=['Return', 'Std', 'Sharp Ratio'])    

    single = pd.concat([weights_result, general_result]).to_frame('Max Sharp Ratio')
    return single

####################### Efficient Frontier (minimize portfolio std)##################
def ef(weights,
       ret, 
       cov_matrix, 
       target_low, 
       target_high,
       rf 
       ):

    target_rets = np.arange( round(target_low*100), round(target_high*100 ) ) / 100

    combine=pd.DataFrame()

    for target_ret in target_rets:

        constraints = [{'type':'eq', 'fun': lambda x: np.sum(x)-1}, {'type':'ineq', 'fun': lambda x: port_ret_annual(x, ret) - target_ret}]

        bounds = [(0,1)]*len(tickers)

        results = minimize(port_std_annual, weights, args=(cov_matrix), method='SLSQP', constraints=constraints, bounds=bounds)

        port_ret = port_ret_annual(results.x, ret)
        port_std = port_std_annual(results.x, cov_matrix)
        sharp = - port_negSharp(results.x, ret, cov_matrix, rf=rf)

        weights_result = pd.Series(results.x, index=tickers)
        general_result = pd.Series([port_ret, port_std, sharp], index=['Return', 'Std', 'Sharp Ratio'])

        single = pd.concat([weights_result, general_result]).to_frame(port_ret)
        combine = pd.concat([combine, single], axis=1)

    return combine
###################################################################################
st.markdown('# Efficient Frontier')


with st.form('ticker_input'):
    
    st.markdown('### Tickers to Optimize')

    col1, col2, col3, col4, col5 = st.columns(5) 

    ticker1 = col1.text_input('Ticker 1')
    ticker2 = col2.text_input('Ticker 2')
    ticker3 = col3.text_input('Ticker 3')
    ticker4 = col4.text_input('Ticker 4')
    ticker5 = col5.text_input('Ticker 5')

    ticker_input = st.form_submit_button('Submit')

tickers = [ticker1, ticker2, ticker3, ticker4, ticker5]
tickers = [i.replace(' ','') for i in tickers]
tickers = list(pd.Series(tickers).replace('',None).dropna())

# Period
end = datetime.today()
start = end - timedelta(days=365*5)

if ticker_input and len(tickers)>1:
# Preparing data from Yahoo
    df = yf.download(tickers, start, end)['Adj Close'] # Dataframe
    tbill = yf.download('^IRX', start, end)['Close'] # Series

    # Specify weights (have to specify tickers to index for later multiple correctly) 
    weights = pd.Series([1/len(tickers)]*len(tickers), index=tickers) # use equal weights as starting point
    weights = weights[df.columns] # to adapt to yahoo finance sorting
    tickers = df.columns # to adapt to yahoo finance sorting

    # handeling return
    rf = tbill.div(252).div(100)
    rf_mean_annual = rf.mean() * 252

    ret = df.pct_change().dropna()
    # log_ret = np.log(ret.add(1))
    excess_ret = ret.apply(lambda x: x - rf).dropna()
    # log_excess_ret = np.log(excess_ret.add(1))

    cov_matrix = ret.cov()
    excess_cov_matrix = excess_ret.cov()
    

    max_sharp = max_sharp_port(weights, ret, cov_matrix, rf=rf_mean_annual)
    
    frontier = ef(weights,
                ret, 
                cov_matrix, 
                target_low=min_std_port(weights, ret, cov_matrix, rf).loc['Return'][0], 
                target_high=max_ret_port(weights, ret, cov_matrix, rf).loc['Return'][0], 
                rf = rf_mean_annual
                )

    slope = (max_sharp.loc['Return'][0] - rf_mean_annual) / max_sharp.loc['Std'][0]

    fig, axes = plt.subplots()
    axes.scatter(frontier.loc['Std'], frontier.loc['Return'])
    axes.set_ylabel('Return')
    axes.set_xlabel('Std')
    axes.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    axes.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))

    axes.scatter(max_sharp.loc['Std'], max_sharp.loc['Return'], marker='*', s=100, c='r')
    axes.annotate('Max Sharp Ratio ', xy=[max_sharp.loc['Std'], max_sharp.loc['Return']-0.01])

    axes.scatter(0, rf_mean_annual, c='r')
    axes.annotate('Risk Free Rate', xy=[0+0.001, rf_mean_annual-0.01])

    axes.plot([0, max_sharp.loc['Std'][0], frontier.loc['Std'].iloc[-1]], 
            [rf_mean_annual, max_sharp.loc['Return'][0], rf_mean_annual+frontier.loc['Std'].iloc[-1]*slope], color='grey', linewidth=1)

    axes.scatter(0, 0, marker='')

    plt.margins(0)

    st.write(max_sharp.style.format('{:.2%}'))
    st.pyplot(axes.get_figure())
    

elif ticker_input and len(tickers)==1:
    st.error('At least 2 inputs')

else:
    pass

################################# General setting##################################
