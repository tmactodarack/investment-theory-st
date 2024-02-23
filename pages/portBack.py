import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib import ticker as mtick
import datetime
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # for better plot quality if we use retina

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

if 'combine' not in st.session_state:
    st.session_state['combine'] = ''

if 'fig' not in st.session_state:
    st.session_state['fig'] = ''

def extract_first_ele(items):
    return [item[0] for item in items]

def extract_sec_ele(items):
    return [item[1] for item in items]
       

        #######################receiving data from html#######################################
def main_calc(start_date, end_date, initial_fund):
    if st.session_state['ticker11'] == '':
        return 'Missing Inputs'

    else: 
        tickers = {}
        weights = {}
        portfolios = {}
        directions = {}
        portfolios_num = 2 
        tickers_num = 5


        for i in range(1, portfolios_num+1):
            tickers = []
            weights = []
            directions = []

            for j in range(1, tickers_num+1):
                tickers.append( st.session_state[f'ticker{i}{j}'] )
                weights.append( st.session_state[f'weight{i}{j}'] )
                directions.append( st.session_state[f'direction{i}{j}'] )
             
            tickers = pd.Series(tickers, name='tickers')
            weights = pd.Series(weights, name='weights').apply(lambda x: float(x) if x !='' else '')
            directions = pd.Series(directions, name='directions')
            directions.replace('Long', 1, inplace=True)
            directions.replace('Short', -1, inplace=True)

            if pd.concat([tickers, weights, directions], axis=1).replace('',None).dropna().reset_index(drop=True).shape[0] != 0:
                portfolios[i] = pd.concat([tickers, weights, directions], axis=1).replace('',None).dropna().reset_index(drop=True)
                portfolios[i].set_index('tickers', inplace=True)


        # for p in range(1, portfolios_num+1):
        #         if len(st.session_state[f'tickers{p}'])==0:
        #             pass
        #         else:
        #             tickers[str(p)] = [s.replace(' ','') for s in st.session_state[f'tickers{p}'].split(',') ]
        #             # st.write(tickers)
                
        #         if len(st.session_state[f'weights{p}']) ==0:
        #             pass
        #         else:
        #             weights[str(p)] = [s.replace(' ','') for s in st.session_state[f'weights{p}'].split(',') ]
        #             # st.write(weights)
                
        #         if len(st.session_state[f'tickers{p}'])==0:
        #             pass
        #         else:
        #             portfolios[str(p)] = {}
                    
        #             for n, t in enumerate(st.session_state[f'tickers{p}']):
        #                 portfolios[str(p)][t] = [ float(weights[str(p)][n]), 1 ]

        #######################receiving data from html##############################################



        ###################setting portfolio here###########################
        period = {
            1:(start_date+pd.offsets.MonthBegin(0), end_date+pd.offsets.MonthEnd(0)),
            2:(start_date+pd.offsets.MonthBegin(0), end_date+pd.offsets.MonthEnd(0)),
            3:(start_date+pd.offsets.MonthBegin(0), end_date+pd.offsets.MonthEnd(0))
        }
        
        initial_fund = {
            1:initial_fund,
            2:initial_fund,
            3:initial_fund
        }

        leverage_multiple = {
            1:1,
            2:1,
            3:1
        }
        ######################################################################


        # Preparing portfolios level dashboard variables
        combine = pd.DataFrame(dtype='float')
        position_list = []
        position_cum_list = pd.DataFrame(dtype='float')
        
        counter = st.empty() ##### Initializing loading counter
        
        for port in portfolios.keys():
            
            counter.write( f':clock{port}: Loading portfolio{port} -- {(port-1)/len(portfolios[port].index):.0%}' ) ##### Used for showing loading

            tickers = list(portfolios[port].index)

            weights = portfolios[port].weights

            short_or_long = portfolios[port].directions

            real_allocation = initial_fund[port] * weights
            leveraged_allocation =  real_allocation * leverage_multiple[port]
            deleverage_adjustment = (short_or_long * leveraged_allocation) * (-1)

            bil = yf.download('^IRX', period[port][0], period[port][1])['Close']
            bil = bil.resample('m').mean().div(12).div(100)

            df =  yf.download(tickers, period[port][0], period[port][1])['Adj Close']
            ret = df.pct_change()

            if len(portfolios[port].index) == 1:    # Fixing the problem that df['Adj Close'] will be a series rather than dataframe issue
                ret = ret.to_frame( list(portfolios[port].index)[0] )
            else:
                pass  

            ############ Tiingo's version, but currently not working on web app ########################
            # bil = tiingo.get_dataframe('BIL',period[port][0], period[port][1], metric_name='adjClose') 
            # bil.index = pd.to_datetime(bil.index).tz_convert(None)
            # bil_mean = bil.pct_change().add(1).resample('m').prod().sub(1).mean()*12

            # df = tiingo.get_dataframe(tickers, period[port][0], period[port][1], metric_name='adjClose')  
            # df.index = pd.to_datetime(df.index).tz_convert(None)
            # ret = df.pct_change()   
            
            ret_month = ret.add(1).resample('m').prod()     # Resampling the ret to monthly ret (add 1)
            
            months = ret_month.index.shape[0]  # Calculating how many months

        ###################################### Main calculation area:######################################################
            
            start_date = ret_month.index[0]    # Get first date

            position = pd.DataFrame(columns=tickers)    # Taking notes of portfolio position change

            for i in ret_month.resample('a').last().index: # Getting rebalace date

                end_date = i

                position_temp = ret_month.loc[start_date: end_date].cumprod() * leveraged_allocation
                
                position_temp = position_temp * short_or_long + deleverage_adjustment + real_allocation 

                position = pd.concat([position, position_temp])


                if i != ret_month.resample('a').last().index[-1]: # Updating allocation and new start date, except for last one

                    real_allocation = position.iloc[-1,:].sum() * weights

                    leveraged_allocation = real_allocation * leverage_multiple[port]
                    
                    deleverage_adjustment = (short_or_long * leveraged_allocation) * (-1)   

                    start_date = ret_month.index[ ret_month.index.to_list().index(i) + 1 ]
                else:
                    pass


            # Preparing results and dashboards
            position_cum = position.sum(axis=1)
            position_cum = pd.concat([position_cum, 
                                    pd.Series(initial_fund[port], 
                                                index=[position_cum.index[0] + pd.DateOffset(months=-1)] 
                                                )
                                    ]).sort_index()

            dd = ((position_cum - position_cum.cummax()) / position_cum.cummax()).min()
            position_mean = position_cum.pct_change().mean() * 12
            position_std = position_cum.pct_change().std() * np.sqrt(12)
            position_excess_return_mean = ( position_cum.pct_change() - bil ).mean() * 12
            position_excess_return_std =  (position_cum.pct_change() - bil ).std() * np.sqrt(12)
            sharp = position_excess_return_mean / position_excess_return_std
            CAGR = ( (position_cum[-1]/position_cum[0]) ** (1/ (months/12)) ) - 1

            single = pd.Series({
                # 'tickers': tickers,
                'Start Date': f'{ret_month.index[0]:%Y-%b}',
                'End Date': f'{ret_month.index[-1]:%Y-%b}',
                'Years': f'{months/12:.2f}',
                'Initial Value': f'{initial_fund[port]:,.0f}',
                'Final Value': f'{position_cum[-1]:,.0f}',
                'CAGR': f'{CAGR:.2%}',
                'Annualized Risk Free Rate': f'{bil.mean()*12:.2%}',
                'Annualized Simple Mean': f'{position_mean:.2%}',
                'Annualized Std': f'{position_std:.2%}',
                'Sharp Ratio': f'{sharp:.2f}',
                'Max Drawdown': f'{dd:.2%}'
            }).to_frame(f'Portfolio {port}')

            position_list.append(position)
            position_cum_list = pd.concat([position_cum_list,
                                        position_cum.to_frame(f'Portfolio {port}: ' + str(tickers)),
                                        ], axis=1)

            combine = pd.concat([combine, single], axis=1)

        axes = position_cum_list.plot(figsize=(12,6))
        # axes.set_title('Portfolio Cumulative Value')
        axes.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        fig = axes.get_figure()
        # fig.patch.set_alpha(0.0)
        # plt.gcf().set_facecolor('red')
        st.session_state['combine'] = combine
        st.session_state['fig'] = fig
        # return (combine, fig)
        counter.empty() ####### clean loading counter

# st.set_page_config(layout="wide")

################## Fast Ideas Function ############################## 
st.markdown('# Portfolio Backtester')
st.markdown('This tool backtests performance of annually rebalanced portfolios. \
            Specify tickers (find tickers at [Yahoo Finance](https://finance.yahoo.com/)) and weights (in the form of decimals and sum to 1)')

tickers = {}
weights = {}
options = ['UTIMCO', 'Half stock half bond', 'Favorite']

def fast():
    if st.session_state['fast_idea'] == 'UTIMCO':
        st.session_state['tickers' + st.session_state['fast_port']] = 'SPY, EFA, TIP, VNQ, GNR, IFRA, AGG, BIL'
        st.session_state['weights'+ st.session_state['fast_port']] = '0.45, 0.15, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05'

    elif st.session_state['fast_idea'] == 'Half stock half bond':
        st.session_state['tickers' + st.session_state['fast_port']] = 'SPY, AGG'
        st.session_state['weights'+ st.session_state['fast_port']] = '0.5, 0.5'

    elif st.session_state['fast_idea'] == 'Favorite':
        st.session_state['tickers' + st.session_state['fast_port']] = 'SPY, EFA, AGG, BIL, INDA, GLD'
        st.session_state['weights'+ st.session_state['fast_port']] = '0.45, 0.05, 0.15, 0.15, 0.1, 0.1'

    else:
        pass

################ General Params ##################################
with st.container(border=True):
    st.markdown('### General Params')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2019-01-01'))
    # datetime.datetime.today()+datetime.timedelta(days=-30.4*60)+pd.offsets.MonthBegin(0)
    with col2:    
        end_date = st.date_input('End Date', value=pd.to_datetime('2023-12-31'))
    # datetime.datetime.today()+pd.offsets.MonthEnd(0)
    with col3:
        initial_fund = st.number_input('Initail Fund ($)', value=10000)
    with col4:
        rebalance = st.selectbox('Rebalance Frequency', options=['Anually'])

    st.markdown('*Note: Start date will always be adjusted to month begin, while end date to month end*')

################## Fast Idea ######################################
# with st.expander(':bulb: Fast Portfolio Idea'):
#     with st.form('fast_form'):
#         col1, col2 = st.columns(2)
#         with col1:
#             st.selectbox('Pick an idea', options=options, key='fast_idea')
#         with col2:
#             st.radio('Paste to portfolio', options=['1', '2', '3'], key='fast_port', horizontal=True)

#         st.form_submit_button('Add', on_click=fast)


#################### Portfolios build up ######################
portfolios_num = 2
tickers_num = 5

with st.form('first_form'):

    cols_port = st.columns(portfolios_num) # building columns for each portfolios

    for i in range(portfolios_num):
        with cols_port[i]:
            st.markdown(f'### Portfolio {i+1}')
            cols_tick = st.columns(3) # building columns inside portfolios
            with cols_tick[0]:
                st.markdown('Ticker')
                for j in range(tickers_num):
                    st.text_input('Ticker', key = f"ticker{i+1}{j+1}", label_visibility='collapsed')
            with cols_tick[1]:
                st.markdown('Weight')
                for j in range(tickers_num):
                    st.text_input('Weight', key = f"weight{i+1}{j+1}", label_visibility='collapsed')
            with cols_tick[2]:
                st.markdown('Direction')
                for j in range(tickers_num):
                    st.selectbox('Direction', options=['Long','Short'], key = f"direction{i+1}{j+1}", label_visibility='collapsed')

    st.markdown('*Check tickers at [Yahoo Finance](https://finance.yahoo.com/)*')
    submit = st.form_submit_button('Submit')



######################## Calculation Area ############################
if not submit:
    pass
else:
    #############Here execute the main calculator##############
    main_calc(start_date, end_date, initial_fund)
    ###########################################################
    with st.container():
        st.markdown('## Metrics')
        st.dataframe(st.session_state['combine'], height=425)

    with st.container():
        st.markdown('## Cumulative Portfolio Value')
        st.pyplot(st.session_state['fig'])




###################################### Below is old code ######################### 
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from matplotlib import pyplot as plt
# plt.style.use('fivethirtyeight')
# from matplotlib import ticker as mtick
# import datetime
# import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # for better plot quality if we use retina


# if 'combine' not in st.session_state:
#     st.session_state['combine'] = ''

# if 'fig' not in st.session_state:
#     st.session_state['fig'] = ''

# def extract_first_ele(items):
#     return [item[0] for item in items]

# def extract_sec_ele(items):
#     return [item[1] for item in items]
       

#         #######################receiving data from html#######################################
# def main_calc(tickers, weights, start_date, end_date, initial_fund):
#     if tickers['1'] == '':
#         pass

#     else: 
#         portfolios = {}
#         directions = {}

#         portfolios_num = 3 
#         tickers_num = 8

#         for p in range(1, portfolios_num+1):
#                 if len(tickers[str(p)])==0:
#                     pass
#                 else:
#                     tickers[str(p)] = [s.replace(' ','') for s in tickers[str(p)].split(',') ]
#                     # st.write(tickers)
                
#                 if len(weights[str(p)]) ==0:
#                     pass
#                 else:
#                     weights[str(p)] = [s.replace(' ','') for s in weights[str(p)].split(',') ]
#                     # st.write(weights)
                
#                 if len(tickers[str(p)])==0:
#                     pass
#                 else:
#                     portfolios[str(p)] = {}
                    
#                     for n, t in enumerate(tickers[str(p)]):
#                         portfolios[str(p)][t] = [ float(weights[str(p)][n]), 1 ]

#         #######################receiving data from html##############################################



#         ###################setting portfolio here###########################
#         period = {
#             '1':(start_date+pd.offsets.MonthBegin(0), end_date+pd.offsets.MonthEnd(0)),
#             '2':(start_date+pd.offsets.MonthBegin(0), end_date+pd.offsets.MonthEnd(0)),
#             '3':(start_date+pd.offsets.MonthBegin(0), end_date+pd.offsets.MonthEnd(0))
#         }
        
#         initial_fund = {
#             '1':initial_fund,
#             '2':initial_fund,
#             '3':initial_fund
#         }

#         leverage_multiple = {
#             '1':1,
#             '2':1,
#             '3':1
#         }
#         ######################################################################


#         # Preparing portfolios level dashboard variables
#         combine = pd.DataFrame(dtype='float')
#         position_list = []
#         position_cum_list = pd.DataFrame(dtype='float')
        
#         counter = st.empty() ##### Initializing loading counter
        
#         for num, port in enumerate(portfolios.keys()):
            
#             counter.write( f':clock{num+1}: Loading portfolio{num+1} -- {(num)/len(portfolios.keys()):.0%}' ) ##### Used for showing loading

#             tickers = list(portfolios[port].keys())

#             weights = extract_first_ele(portfolios[port].values())
#             weights = pd.Series(weights, index=tickers)

#             short_or_long = extract_sec_ele(portfolios[port].values()) 
#             short_or_long = pd.Series(short_or_long, index=tickers)

#             real_allocation = initial_fund[port] * weights
#             leveraged_allocation =  real_allocation * leverage_multiple[port]
#             deleverage_adjustment = (short_or_long * leveraged_allocation) * (-1)

#             bil = yf.download('^IRX', period[port][0], period[port][1])['Close']
#             bil = bil.resample('m').mean().div(12).div(100)

#             df =  yf.download(tickers, period[port][0], period[port][1])['Adj Close']
#             ret = df.pct_change()

#             if len(portfolios[port].keys()) == 1:    # Fixing the problem that df['Adj Close'] will be a series rather than dataframe issue
#                 ret = ret.to_frame( list(portfolios[port].keys())[0] )
#             else:
#                 pass  

#             ############ Tiingo's version, but currently not working on web app ########################
#             # bil = tiingo.get_dataframe('BIL',period[port][0], period[port][1], metric_name='adjClose') 
#             # bil.index = pd.to_datetime(bil.index).tz_convert(None)
#             # bil_mean = bil.pct_change().add(1).resample('m').prod().sub(1).mean()*12

#             # df = tiingo.get_dataframe(tickers, period[port][0], period[port][1], metric_name='adjClose')  
#             # df.index = pd.to_datetime(df.index).tz_convert(None)
#             # ret = df.pct_change()   
            
#             ret_month = ret.add(1).resample('m').prod()     # Resampling the ret to monthly ret (add 1)
            
#             months = ret_month.index.shape[0]  # Calculating how many months

#         ###################################### Main calculation area:######################################################
            
#             start_date = ret_month.index[0]    # Get first date

#             position = pd.DataFrame(columns=tickers)    # Taking notes of portfolio position change

#             for i in ret_month.resample('a').last().index: # Getting rebalace date

#                 end_date = i

#                 position_temp = ret_month.loc[start_date: end_date].cumprod() * leveraged_allocation
                
#                 position_temp = position_temp * short_or_long + deleverage_adjustment + real_allocation 

#                 position = pd.concat([position, position_temp])


#                 if i != ret_month.resample('a').last().index[-1]: # Updating allocation and new start date, except for last one

#                     real_allocation = position.iloc[-1,:].sum() * weights

#                     leveraged_allocation = real_allocation * leverage_multiple[port]
                    
#                     deleverage_adjustment = (short_or_long * leveraged_allocation) * (-1)   

#                     start_date = ret_month.index[ ret_month.index.to_list().index(i) + 1 ]
#                 else:
#                     pass


#             # Preparing results and dashboards
#             position_cum = position.sum(axis=1)
#             position_cum = pd.concat([position_cum, 
#                                     pd.Series(initial_fund[port], 
#                                                 index=[position_cum.index[0] + pd.DateOffset(months=-1)] 
#                                                 )
#                                     ]).sort_index()

#             dd = ((position_cum - position_cum.cummax()) / position_cum.cummax()).min()
#             position_mean = position_cum.pct_change().mean() * 12
#             position_std = position_cum.pct_change().std() * np.sqrt(12)
#             position_excess_return_mean = ( position_cum.pct_change() - bil ).mean() * 12
#             position_excess_return_std =  (position_cum.pct_change() - bil ).std() * np.sqrt(12)
#             sharp = position_excess_return_mean / position_excess_return_std
#             CAGR = ( (position_cum[-1]/position_cum[0]) ** (1/ (months/12)) ) - 1

#             single = pd.Series({
#                 # 'tickers': tickers,
#                 'Start Date': f'{ret_month.index[0]:%Y-%b}',
#                 'End Date': f'{ret_month.index[-1]:%Y-%b}',
#                 'Years': f'{months/12:.2f}',
#                 'Initial Value': f'{initial_fund[port]:,.0f}',
#                 'Final Value': f'{position_cum[-1]:,.0f}',
#                 'CAGR': f'{CAGR:.2%}',
#                 'Annualized Risk Free Rate': f'{bil.mean()*12:.2%}',
#                 'Annualized Simple Mean': f'{position_mean:.2%}',
#                 'Annualized Std': f'{position_std:.2%}',
#                 'Sharp Ratio': f'{sharp:.2f}',
#                 'Max Drawdown': f'{dd:.2%}'
#             }).to_frame('Portfolio ' + port)

#             position_list.append(position)
#             position_cum_list = pd.concat([position_cum_list,
#                                         position_cum.to_frame('Portfolio ' + port + ' ' + str(tickers)),
#                                         ], axis=1)

#             combine = pd.concat([combine, single], axis=1)

#         axes = position_cum_list.plot(figsize=(12,6))
#         # axes.set_title('Portfolio Cumulative Value')
#         axes.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
#         fig = axes.get_figure()
#         # fig.patch.set_alpha(0.0)
#         # plt.gcf().set_facecolor('red')
#         st.session_state['combine'] = combine
#         st.session_state['fig'] = fig
#         # return (combine, fig)
#         counter.empty() ####### clean loading counter

# # st.set_page_config(layout="wide")

# ##################Function of adding portfolio############################## 
# st.markdown('# Portfolio Backtester')
# st.markdown('This tool produces backtest results of annual rebalancing portfolio with specified equity tickers and weights\
#             (Use comma to seprate inputs). The data source is from [Yahoo Finance](https://finance.yahoo.com/); check tickers there.')
# st.markdown('You also can try out some built-in portfolio ideas from the fast ideas dropdown.')

# tickers = {}
# weights = {}
# options = ['UTIMCO', 'Half stock half bond', 'Favorite']

# def fast():
#     if st.session_state['fast_idea'] == 'UTIMCO':
#         st.session_state['tickers' + st.session_state['fast_port']] = 'SPY, EFA, TIP, VNQ, GNR, IFRA, AGG, BIL'
#         st.session_state['weights'+ st.session_state['fast_port']] = '0.45, 0.15, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05'

#     elif st.session_state['fast_idea'] == 'Half stock half bond':
#         st.session_state['tickers' + st.session_state['fast_port']] = 'SPY, AGG'
#         st.session_state['weights'+ st.session_state['fast_port']] = '0.5, 0.5'

#     elif st.session_state['fast_idea'] == 'Favorite':
#         st.session_state['tickers' + st.session_state['fast_port']] = 'SPY, EFA, AGG, BIL, INDA, GLD'
#         st.session_state['weights'+ st.session_state['fast_port']] = '0.45, 0.05, 0.15, 0.15, 0.1, 0.1'

#     else:
#         pass

# ################Handling Sidebar here##################################
# with st.sidebar:
#     st.markdown('# Additional Params')

#     start_date = st.date_input('Start Date', value=pd.to_datetime('2019-01-01'))
#     # datetime.datetime.today()+datetime.timedelta(days=-30.4*60)+pd.offsets.MonthBegin(0)
#     end_date = st.date_input('End Date', value=pd.to_datetime('2023-12-31'))
#     # datetime.datetime.today()+pd.offsets.MonthEnd(0)
#     st.markdown('###### *Note: Start date and end date will always be adjusted to month begin, and month end*')
    
#     initial_fund = st.number_input('Initail Fund ($)', value=10000)

# with st.expander(':bulb: Fast Portfolio Idea'):
#     with st.form('fast_form'):
#         col1, col2 = st.columns(2)
#         with col1:
#             st.selectbox('Pick an idea', options=options, key='fast_idea')
#         with col2:
#             st.radio('Paste to portfolio', options=['1', '2', '3'], key='fast_port', horizontal=True)

#         st.form_submit_button('Add', on_click=fast)


# ############ Main form button form here######################
# with st.form('first_form'):
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.markdown('### Portfolio 1')
#         tickers['1'] = st.text_input('Tickers1', placeholder='e.g., SPY, AGG', key='tickers1')
#         weights['1'] = st.text_input('Weights1', placeholder='e.g., 0.5, 0.5', key='weights1')
    
#     with col2:
#         st.markdown('### Portfolio 2')
#         tickers['2'] = st.text_input('Tickers2', key='tickers2')
#         weights['2'] = st.text_input('Weights2', key='weights2')

#     with col3:
#         st.markdown('### Portfolio 3')
#         tickers['3'] = st.text_input('Tickers3', key='tickers3')
#         weights['3'] = st.text_input('Weights3', key='weights3')

#     submit = st.form_submit_button('Submit')

# if not submit:
#     pass
# else:
#     #############Here execute the main calculator##############
#     main_calc(tickers,weights, start_date, end_date, initial_fund)
#     ###########################################################
#     with st.container():
#         st.markdown('## Metrics')
#         st.dataframe(st.session_state['combine'], height=425)

#     with st.container():
#         st.markdown('## Cumulative Portfolio Value')
#         st.pyplot(st.session_state['fig'])

