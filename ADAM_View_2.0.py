""" ADAM Visions """

import streamlit as stl
import yfinance
from plotly import graph_objects as go
from datetime import date
import pandas as pd

if True:
    def incredible_data_requestor_full(stockz, time_period, n, d=5, pr='1d'):
        import datetime as DT
        import time
        dff = []
        loop = n
        i = 1
        while i < (loop + 1):
            today = DT.date.today()
            endDate = today - DT.timedelta(days=d * (i - 1))
            startDate = today - DT.timedelta(days=d * i)
            df = yfinance.download(stockz, interval=time_period, start=startDate, end=endDate)
            if i == 1:
                df = yfinance.download(stockz, interval=time_period, period=pr, prepost=True)
            print(endDate)
            print(startDate)
            print(df)
            df = df[::-1]
            dff.extend(df)
            time.sleep(1)
            print('cicle = ', i)
            i += 1

        return df

    def EMA(x, n):
        import numpy as np
        """
        returns an n period exponential moving average for
        the time series s

        s is a list ordered from oldest (index 0) to most
        recent (index -1)
        n is an integer

        returns a numeric array of the exponential
        moving average
        """
        x = np.array(x)
        ema = []
        j = 1

        # get n sma first and calculate the next n period ema
        sma = sum(x[:n]) / n
        multiplier = 2 / float(1 + n)
        ema.append(sma)

        # EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
        ema.append(((x[n] - sma) * multiplier) + sma)

        # now calculate the rest of the values
        for i in x[n + 1:]:
            tmp = ((i - ema[j]) * multiplier) + ema[j]
            j = j + 1
            ema.append(tmp)
        ema = ema[::-1]

    def SuperTrend(df, lb=14, mlt=3.8):
        import numpy as np

        def get_supertrend(high, low, close, lookback, multiplier):
            # ATR
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.ewm(lookback).mean()
            # H/L AVG AND BASIC UPPER & LOWER BAND
            hl_avg = (high + low) / 2
            upper_band = (hl_avg + multiplier * atr).dropna()
            lower_band = (hl_avg - multiplier * atr).dropna()
            # FINAL UPPER BAND
            final_bands = pd.DataFrame(columns=['upper', 'lower'])
            final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
            final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
            for i in range(len(final_bands)):
                if i == 0:
                    final_bands.iloc[i, 0] = 0
                else:
                    if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (close[i - 1] > final_bands.iloc[i - 1, 0]):
                        final_bands.iloc[i, 0] = upper_band[i]
                    else:
                        final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]
            # FINAL LOWER BAND
            for i in range(len(final_bands)):
                if i == 0:
                    final_bands.iloc[i, 1] = 0
                else:
                    if (lower_band[i] > final_bands.iloc[i - 1, 1]) | (close[i - 1] < final_bands.iloc[i - 1, 1]):
                        final_bands.iloc[i, 1] = lower_band[i]
                    else:
                        final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]
            # SUPERTREND
            supertrend = pd.DataFrame(columns=[f'supertrend_{lookback}'])
            supertrend.iloc[:, 0] = [x for x in final_bands['upper'] - final_bands['upper']]
            for i in range(len(supertrend)):
                if i == 0:
                    supertrend.iloc[i, 0] = 0
                elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] < final_bands.iloc[i, 0]:
                    supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
                elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] > final_bands.iloc[i, 0]:
                    supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
                elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] > final_bands.iloc[i, 1]:
                    supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
                elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] < final_bands.iloc[i, 1]:
                    supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
            supertrend = supertrend.set_index(upper_band.index)
            supertrend = supertrend.dropna()[1:]
            # ST UPTREND/DOWNTREND
            upt = []
            dt = []
            close = close.iloc[len(close) - len(supertrend):]
            for i in range(len(supertrend)):

                if close[i] > supertrend.iloc[i, 0]:
                    upt.append(supertrend.iloc[i, 0])
                    dt.append(np.nan)
                elif close[i] < supertrend.iloc[i, 0]:
                    upt.append(np.nan)
                    dt.append(supertrend.iloc[i, 0])
                else:
                    upt.append(np.nan)
                    dt.append(np.nan)
            st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
            upt.index, dt.index = supertrend.index, supertrend.index
            return st, upt, dt

        def implement_st_strategy(prices, st):
            buy_price = []
            sell_price = []
            st_signal = []
            signal = 0

            for i in range(len(st)):
                if st[i - 1] > prices[i - 1] and st[i] < prices[i]:
                    if signal != 1:
                        buy_price.append(prices[i])
                        sell_price.append(np.nan)
                        signal = 1
                        st_signal.append(signal)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        st_signal.append(0)
                elif st[i - 1] < prices[i - 1] and st[i] > prices[i]:
                    if signal != -1:
                        buy_price.append(np.nan)
                        sell_price.append(prices[i])
                        signal = -1
                        st_signal.append(signal)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        st_signal.append(0)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    st_signal.append(0)

            return buy_price, sell_price, st_signal

        print(df['Close'])
        df_st = get_supertrend(df['High'], df['Low'], df['Close'], lookback=lb, multiplier=mlt)
        # print(df_st)

        buy_price, sell_price, st_signal = implement_st_strategy(df['Close'], df_st[0])

        # print(buy_price, sell_price, st_signal)
        return df_st, buy_price, sell_price, st_signal

    def ATR(df, mult):
        """the DF need to be Full"""
        import numpy as np
        data = df
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(mult).sum() / mult
        atr = list(atr)

        return atr

    def ATR_SL(df, mult):
        """the DF need to be Full"""
        import numpy as np
        data = df
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = EMA(true_range, mult)
        atr = list(atr)

        return atr

    def TMA(df, ws):
        df = df
        df = pd.DataFrame(df)
        print(df)
        # df['mean0']=df.mean(0)
        df = df.mean(1)
        df = df.rolling(window=ws, center=False).mean()
        return df

    def MA(x, ws):
        """
        :param x: una lista di numeri
        :param ws: la lunghezza
        :return: una lista
        """
        import numpy as np
        # x = x[::-1]
        import pandas as pd
        window_sizes = ws
        series = pd.Series(x)
        windows = series.rolling(window_sizes)
        MA = windows.mean()
        MA_list = MA.tolist()
        FMA = MA_list[window_sizes - 1:]
        FMA = list(FMA)
        for i in range(ws - 1): FMA.insert(i, x[i])
        FMA = FMA[::-1]

        return FMA

START = "2022-10-01"
END = date.today().strftime("%Y-%m-%d")

stl.header('Perla Enviroment ADAM View 2.0')
stock = str(stl.text_input('Stock'))
time_period = str(stl.text_input('TimeFrame (5m, 15m, 1d, 1mo)'))
tipe = str(stl.text_input('Time Period (1d, 1w, 1mo)'))

dffstart = yfinance.download(tickers=stock, start=START, end=END,  interval=time_period)
df = dffstart.Close
dff = dffstart
dff.reset_index(inplace=True)

showframe = stl.checkbox('Show DataFrame?')
if showframe:
    stl.subheader('Data Frame')
    stl.dataframe(dff)

Filtring = stl.checkbox('Usare Filtro?')
MA1se = stl.checkbox('ad MA 1?')
MA2se = stl.checkbox('ad MA 2?')
Stse = stl.checkbox("Super Trend?")
if Stse:
    EtichetteST = stl.checkbox('Etichette ST?')
    dffull = incredible_data_requestor_full(stock, time_period, 1, pr=tipe)
#ATRse = stl.checkbox("double ATR?")
TMAse = stl.checkbox("TMAs?")
if MA1se:
    MA1_Lenght = int(stl.slider("adMA short Lenght", 1, 200))
    MA1 = MA(df, MA1_Lenght)[::-1]
if MA2se:
    MA2_Lenght = int(stl.slider("adMA2 long Lenght", 1, 200))
    MA2 = MA(df, MA2_Lenght)[::-1]
if Stse:
    ST_M = int(stl.slider("ST period", 1, 60))
    ST_P = float(stl.slider("ST Multiplier", 1.0, 6.0))
    df_st, buy_price, sell_price, st_signal = SuperTrend(dffull, ST_P, ST_M)
if TMAse:
    TMAs_Lenght = int(stl.slider('TMA Lenght', 1, 200))
    TMA = TMA(df, TMAs_Lenght)
    TMA = list(TMA)
    TMADOWN = []
    TMAUP = []
    ATR_PERIOD = 120
    ATR_MULT = 1.8
    atr = ATR(dff, ATR_PERIOD)
    rangee = []
    for i in range(len(atr)):
        rangee.append(atr[i] * ATR_MULT)

    for i in range(len(TMA)):
        TMADOWN.append(TMA[i] - rangee[i])
        TMAUP.append(TMA[i] + rangee[i])

stl.subheader('Plot')
stl.write('Puoi ingrandire il grafico')

if Filtring:
    MA3_Lenght = int(stl.slider("adMA filter Lenght", 1, 200))
    MA3 = MA(df, MA3_Lenght)[::-1]

if ATRse:
    x1 = ATR_SL(dffull, 9)
    x2 = ATR_SL(dffull, 9)
    xx1 = []
    xx2 = []
    for i in range(len(x1)):
        xx1.append((x1[i] * 1.5) + dff['High'][i])
        xx2.append(dff['Low'][i] - (x1[i] * 1.5))

    xx1.extend(df[0:8])
    xx2.extend(df[0:8])

if True:
    fig1 = go.Figure()
    fig1.add_trace(go.Ohlc(x=dff['Datetime'],
                        open=dff['Open'],
                        high=dff['High'],
                        low=dff['Low'],
                        close=dff['Close']))

if MA1se:
    fig1.add_trace(go.Scatter(x=dff['Datetime'], y=MA1, name='MA1'))
if MA2se:
    fig1.add_trace(go.Scatter(x=dff['Datetime'], y=MA2, name='MA2'))
if Filtring:
    fig1.add_trace(go.Scatter(x=dff['Datetime'], y=MA3, name='MA3'))
if ATRse:
    fig1.add_scatter(x=dff['Datetime'], y=xx1, line=dict(color='red', width=1), name='ATR Up')
    fig1.add_scatter(x=dff['Datetime'], y=xx2, line=dict(color='green', width=1), name='ATR Down')
if Stse:
    fig1.add_scatter(x=dffull.index, y=df_st[0], line=dict(color='red', width=1), name='Super Trend Up')
    fig1.add_scatter(x=dffull.index, y=df_st[2], line=dict(color='green', width=1), name='Super Trend down')
    if EtichetteST:
        fig1.add_trace(go.Scatter(
            x=dffull.index,
            y=buy_price,
            mode="markers+text",
            name="Buy Entris",
            text='SELL',
            textposition="top center"))
        fig1.add_trace(go.Scatter(
            x=dffull.index,
            y=sell_price,
            mode="markers+text",
            name="Sell Entries",
            text='BUY',
            textposition="top center"))
if TMAse:
    fig1.add_scatter(x=dff['Datetime'], y=TMAUP, line=dict(color='black', width=1), name='TMA Up')
    fig1.add_scatter(x=dff['Datetime'], y=TMADOWN, line=dict(color='black', width=1), name='TMA Down')

fig1.update_xaxes(fixedrange=False)
fig1.update_yaxes(fixedrange=False)

stl.plotly_chart(fig1)


