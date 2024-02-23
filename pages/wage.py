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
            Wage inflation is a crucial metrics for Fed to gauge potential core price pressure. 
            Here we focus on employment cost index(ECI) and average hourly earning(AHE).
            ECI is released by quater while AHE is released by month. When comparing ECI, AHE, and core PCE, all are adjusted to quaterly number.
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
                    # legend = dict(orientation='h', y=1.0, x=1, xanchor='right', yanchor='bottom'),
                    margin=dict(b=50,l=70,r=100,t=70),
                    yaxis=dict(tickformat='.1%')
                    )
st.plotly_chart(longTerm_fig, use_container_width=True)
#######################################


###### ECI QoQ ######
st.markdown('### ECI QoQ')


start='2022'
end = datetime.datetime.today() + datetime.timedelta(days=100)

eci_qoq = combine['Employment Cost Index'].dropna().pct_change().loc[start:end]
title = 'Employment Cost Index QoQ'

fig = go.Figure()
fig.add_trace(go.Bar(x=eci_qoq.index, y=eci_qoq, name='ECI QoQ'))
fig.add_trace(go.Scatter(x=[eci_qoq.index[-1]], y=[eci_qoq[-1]], mode='markers',marker_symbol='star',marker_size=11 , name='Update: '+f"{updates[-1]:%m/%d/%y}"))
# fig.add_annotation(x=test.index[-1], y=test[-1], text='Latest Release')
fig.update_layout(template='ggplot2', 
                  showlegend=True,
                #   title_text=title,
                  title_x=0.1,
                  legend = dict(orientation='h', y=1.02, x=1, xanchor='right'),
                #   margin=dict(b=50,l=70,r=70,t=70),
                  xaxis=dict(
    tickvals = eci_qoq.index, 
    ticktext = eci_qoq.index.to_period('q').to_series().astype(str)
    ), yaxis=dict(
        tickformat='.1%')
                )
with st.container():
    st.plotly_chart(fig)