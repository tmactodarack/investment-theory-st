import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib import ticker as mtick
import datetime
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # for better plot quality if we use retina

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


st.markdown('# Wage Inflation')
st.markdown('''
            Wage inflation is a crucial metrics for Fed to **gauge potential core price pressure**. 
            Here we focus on employment cost index(ECI) and average hourly earning(AHE).
            ECI is released by quater while AHE is released by month. *When comparing ECI, AHE, and core PCE, all are adjusted to quaterly number.*  
            ***
            '''
            )


################################ Fetch data ######################################################
# API key for FRED 
api_key = '&api_key=82309bb88c2901b87dc6e40182e47496'
tickers = {'ECIWAG': 'Employment Cost Index', # ECIWAG is SA, using NSA for yoy might be wrong
           'CES0500000003': 'Average Hourly Earnings', # CES0500000003 is SA
           'PCEPILFE': 'Core PCE' # PCEPILFE is SA
           } 

combine = pd.DataFrame()
updates = pd.Series(dtype='float')

for i in tickers.keys(): 
    url = f'https://api.stlouisfed.org/fred/series?series_id={i}{api_key}' # get latest release date
    df = pd.read_xml(url, parse_dates=['last_updated'])
    updates[tickers[i]] = (df['last_updated'][0])

    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={i}{api_key}' # get data
    df = pd.read_xml(url, parse_dates=['date'])
    df.set_index('date', inplace=True)
    filt = (df['value'] != '.') # some data from fred is in weird . string
    single = df[filt]['value'].apply(lambda x: float(x)).to_frame(tickers[i]) # excluding . and turn into float
    combine = pd.concat([combine, single], axis=1)
    combine.index.name = None
##################################################################################################


############ Long trend ###############
longTerm = combine[['Employment Cost Index', 'Average Hourly Earnings', 'Core PCE']].resample('q').mean().dropna().pct_change(4)
longTerm_fig = go.Figure()
longTerm_fig.add_trace(go.Line(x=longTerm.index, y=longTerm['Employment Cost Index'], name='ECI YoY'))
longTerm_fig.add_trace(go.Line(x=longTerm.index, y=longTerm['Average Hourly Earnings'], name='AHE YoY'))
longTerm_fig.add_trace(go.Line(x=longTerm.index, y=longTerm['Core PCE'], name='Core PCE YoY'))
longTerm_fig.update_layout(template='ggplot2', 
                    showlegend=True,
                    title=dict(text='Long Term Relationship', x=0.03, y=0.9, font=dict(size=30)),
                    # title_x=0.1,
                    legend = dict(orientation='h', y=1.0, x=1, xanchor='right', yanchor='bottom'),
                    margin=dict(b=50,l=70,r=70,t=70),
                    yaxis=dict(tickformat='.1%')
                    )
st.plotly_chart(longTerm_fig, use_container_width=True)
st.markdown('''
            The core inflation is not highly tied to wage inflation, or to put it more generally, 
            they may be affected by a more high level macro environment. For ECI and ACH, they may share somewhat similar
            long-term trend like 2010 to 2020 they both ticked up gradually; however, for short term movement they are very noisy during this period.
            Especially in 2020 after COVID, AHE jumped while ECI dipped, reflecting AHE does not track the same employees components while
            ECI does. Speaking back to the relationship between core PCE and wage inflation, we can see n at least i2013 - 2015 and 2019-2020, they
            moved in the opposite way which may be counter intuition.  
            ***
            ''')

################ ECI QoQ ######################
start='2022'
end = datetime.datetime.today() + datetime.timedelta(days=100)

eci_qoq = combine['Employment Cost Index'].dropna().pct_change().loc[start:end]
title = 'Employment Cost Index QoQ'

fig = go.Figure()
fig.add_trace(go.Bar(x=eci_qoq.index, y=eci_qoq, name='ECI QoQ'))
fig.add_trace(go.Scatter(x=[eci_qoq.index[-1]], 
                         y=[eci_qoq[-1]], 
                         mode='markers',
                         marker_symbol='star',
                         marker_size=11 , 
                         name='Update: ' + f"{updates['Employment Cost Index']:%m/%d/%y}"))
# fig.add_annotation(x=test.index[-1], y=test[-1], text='Latest Release')
fig.update_layout(template='ggplot2', 
                  showlegend=True,
                  title=dict(text=title, x=0.03, y=0.9, font=dict(size=30)),
                #   title_x=0.1,
                #   legend = dict(orientation='h', y=1.02, x=1, xanchor='right'),
                  margin=dict(b=50,l=70,r=70,t=70),
                  xaxis=dict(tickvals = eci_qoq.index, ticktext = eci_qoq.index.to_period('q').to_series().astype(str), fixedrange=True), 
                  yaxis=dict(tickformat='.1%', fixedrange=True)
                )

with st.container():
    st.plotly_chart(fig, use_container_width=True)

with st.expander(' :bulb: Tips'):
    st.markdown('''
            We'd like to check whether there is insight or a trend for ECI QoQ based on:  
            - Whther the latest number is 1 year high or low  
            - Whether there is 3 or more consecutive increase or decrease  
            ''')
st.markdown('***')

################ ECI YoY ######################
start='2022'
end = datetime.datetime.today()
eci_yoy = combine['Employment Cost Index'].dropna().pct_change(4).loc[start:end]

eci_yoy_fig = go.Figure()
eci_yoy_fig.add_trace(go.Line(x=eci_yoy.index, y=eci_yoy, name='ECI YoY'))
eci_yoy_fig.add_trace(go.Scatter(x=[eci_yoy.index[-1]], 
                         y=[eci_yoy[-1]], 
                         mode='markers',
                         marker_symbol='star',
                         marker_size=11, 
                         name='Update: '+f"{updates['Employment Cost Index']:%m/%d/%y}"))
eci_yoy_fig.update_layout(template='ggplot2', 
                    showlegend=True,
                    title=dict(text='Employment Cost Index YoY', x=0.03, y=0.9, font=dict(size=30)),
                    # title_x=0.1,
                    legend = dict(orientation='h', y=1.0, x=1, xanchor='right', yanchor='bottom'),
                    margin=dict(b=50,l=70,r=70,t=70),
                    xaxis=dict(tickvals = eci_yoy.index, ticktext = eci_yoy.index.to_period('q').to_series().astype(str)), 
                    yaxis=dict(tickformat='.1%')
                    )
st.plotly_chart(eci_yoy_fig, use_container_width=True)