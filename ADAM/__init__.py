import pandas as pd
import yfinance


def incredible_data_requestor_b(stockz, time_period, rep):
    import requests
    import json
    import datetime as DT
    import pandas as pd
    import time
    dff = []
    loop = rep
    i = 1
    print('attivo funzione raccolta dati Laggati')
    while i < (loop + 1):
        today = DT.date.today() + DT.timedelta(days=1)
        endDate = today - DT.timedelta(days=60 * (i - 1))
        startDate = today - DT.timedelta(days=60 * i)
        if i == 1:
            endDate = today
            startDate = today - DT.timedelta(days=60)
        df = yfinance.download(stockz, interval=time_period, start=startDate, end=endDate)
        # if i == 1:
        #   df = yfinance.download(stockz, interval=time_period, period='1mo', prepost=True)
        print(endDate)
        print(startDate)
        print(df)
        df = df['Close']
        df = df[::-1]
        if i == 1:
            df = df[1:]

        dff.extend(df)
        # print('dff = ', dff)
        print('cicle = ', i)
        i += 1

    return dff


def incredible_data_requestor_min_full(stockz, time_period, rep):
    import datetime as DT
    import time
    dff = []
    loop = rep
    i = 1
    while i < (loop + 1):
        today = DT.date.today()
        endDate = today - DT.timedelta(days=5 * (i - 1))
        startDate = today - DT.timedelta(days=5 * i)
        df = yfinance.download(stockz, interval=time_period, start=startDate, end=endDate)
        if i == 1:
            df = yfinance.download(stockz, interval=time_period, period='1mo', prepost=True)
        print(endDate)
        print(startDate)
        print(df)
        df = df
        dff.extend(df)
        # print('dff = ', dff)
        time.sleep(1)
        print('cicle = ', i)
        i += 1

    return df


def incredible_data_requestor_c(stockz, time_period, rep):
    import datetime as DT
    import time
    dff = []
    loop = rep
    i = 1

    df = yfinance.download(stockz, interval=time_period, period='1d', prepost=True)

    print(df)
    df = df['Close']
    df = df[::-1]
    dff.extend(df)
    # print('dff = ', dff)
    time.sleep(1)
    print('cicle = ', i)
    i += 1

    return dff


def incredible_data_requestor_d(stockz, time_period, rep, n=300, t=0.2):
    import datetime as DT
    import time
    dff = []
    loop = rep
    i = 1

    while i < (loop + 1):
        today = DT.date.today()
        endDate = today - DT.timedelta(days=n * (i - 1))
        startDate = today - DT.timedelta(days=n * i)
        df = yfinance.download(stockz, interval=time_period, start=startDate, end=endDate)
        if i == 1:
            # df = yfinance.download(stockz, interval=time_period, period='2mo', prepost=
            df = yfinance.download(stockz, interval=time_period, start=startDate, end=endDate)
        print(endDate)
        print(startDate)
        print(df)
        df = df['Close']
        # df = df[::-1]
        dff.extend(df)
        # print('dff = ', dff)
        # time.sleep(1)
        print('cicle = ', i)
        i += 1
        time.sleep(t)

    return dff


def get_data(type):
    data = yfinance.download(tickers=type, period='5m', interval='1m')

    """print(data)
    print(data['Adj Close'][0])"""
    return data['Adj Close'][0]


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


def incredible_data_requestor_b_Oper(stockz, time_period, rep):
    import requests
    import json
    import datetime as DT
    import pandas as pd
    import time
    dff = []
    loop = rep
    i = 1
    print('attivo funzione raccolta dati Laggati')
    while i < (loop + 1):
        today = DT.date.today() + DT.timedelta(days=1)
        endDate = today - DT.timedelta(days=5 * (i - 1))
        startDate = today - DT.timedelta(days=5 * i)
        if i == 1:
            endDate = today
            startDate = today - DT.timedelta(days=5)
        df = yfinance.download(stockz, interval=time_period, start=startDate, end=endDate)
        # if i == 1:
        #   df = yfinance.download(stockz, interval=time_period, period='1mo', prepost=True)
        print(endDate)
        print(startDate)
        print(df)
        df = df['Open']
        df = df[::-1]
        if i == 1:
            df = df[1:]

        dff.extend(df)
        # print('dff = ', dff)
        time.sleep(1)
        print('cicle = ', i)
        i += 1

    return dff


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

    df_st = get_supertrend(df['High'], df['Low'], df['Close'], lookback=lb, multiplier=mlt)
    #print(df_st)

    buy_price, sell_price, st_signal = implement_st_strategy(df['Close'], df_st[0])

    # print(buy_price, sell_price, st_signal)
    return df_st, buy_price, sell_price, st_signal


def TMA(df, ws):
    df = df
    df = pd.DataFrame(df)
    print(df)
    # df['mean0']=df.mean(0)
    df = df.mean(1)
    df = df.rolling(window=ws, center=False).mean()
    return df


def MWMA(x, ws, mean):
    ma = MA(x, ws)
    fma = []
    for i in range(len(ma)):
        fma.append((ma[i] + mean[i]) / 2)
    return fma


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
    for i in range(ws - 1):
        FMA.insert(i, x[i])
    FMA = FMA[::-1]

    return FMA


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

    return ema


def ADAM(MA1, MA2):
    if MA1[-1] == MA2[-1]:
        if MA1[-2] > MA2[-2]:
            return 'sell'
        elif MA1[-2] < MA2[-2]:
            return 'buy'
    else:
        print('not buy point here')
        return 'NaN'


def variance(df):
    varf = []
    for i in range(len(df)):
        if i > 0:
            var = df[i] - df[i - 1]
            var = np.absolute(var / df[i - 1] * 100)
            varf.append(var)
        else:
            varf.append(0.00001)

    print('varianza media = ', np.average(varf))
    return varf


def ADAM_adaptive(MA1, MA2, ws=3):
    ws = ws
    s = 'sell'
    b = 'buy'
    NaN = 'NaN'
    # print('ws = ', ws)
    if MA1[-ws] > MA1[-1]:
        if MA1[-ws] > MA2[-ws] and MA1[-1] < MA2[-1]:
            # print('adsell')
            f = s
        else:
            # print('adNaN after trying sell')
            f = NaN
    elif MA1[-ws] < MA1[-1]:
        if MA1[-ws] < MA2[-ws] and MA1[-1] > MA2[-1]:
            # print('adbuy')
            f = b
        else:
            # print('adNaN after trying buy')
            f = NaN

    else:
        # print('adNaN')
        f = NaN

    return f


def ADAM_adaptive_precise(MA1, MA2, ws=3):
    ws = ws
    s = 'sell'
    b = 'buy'
    NaN = 'NaN'
    print('ws = ', ws)
    if MA1[-ws] > MA1[-1]:
        if MA1[-ws] > MA2[-ws] and MA1[-1] < MA2[-1]:
            print('adsell')
            f = s
        else:
            print('adNaN after trying sell')
            f = NaN
    elif MA1[-ws] < MA1[-1]:
        if MA1[-ws] < MA2[-ws] and MA1[-1] > MA2[-1]:
            print('adbuy')
            f = b
        else:
            print('adNaN after trying buy')
            f = NaN

    else:
        print('adNaN')
        f = NaN

    return f


def ADAM_adaptive_for_Stoploss(MA1, MA2, action, ws=30):
    ws = ws
    s = 'sell'
    b = 'buy'
    NaN = 'NaN'
    print('ws = ', ws)
    if MA1[-ws] > MA1[-1]:
        if MA1[-ws] > MA2[-ws] and MA1[-1] < MA2[-1]:
            print('adsell')
            f = s
        else:
            print('adNaN after trying sell')
            f = NaN
    elif MA1[-ws] < MA1[-1]:
        if MA1[-ws] < MA2[-ws] and MA1[-1] > MA2[-1]:
            print('adbuy')
            f = b
        else:
            print('adNaN after trying buy')
            f = NaN

    else:
        print('adNaN')
        f = NaN

    if f == action:
        tp = 100
        sp = 450
    else:
        tp = 30
        sp = 150

    return tp, sp


def graphic_MA(x, MA1, MA2):
    # x = x[::-1]
    import matplotlib.pyplot as plt
    plt.plot(x, color='blue', marker='o')
    plt.plot(MA1, color='green')
    plt.plot(MA2, color='red')
    plt.show()


def suona():
    import winsound as ws
    ws.Beep(500, 50)
    ws.Beep(1000, 1000)
    time.sleep(0.5)
    ws.Beep(500, 50)
    ws.Beep(1000, 1000)
    time.sleep(1)
    ws.Beep(500, 50)
    ws.Beep(1000, 1000)
    time.sleep(0.5)
    ws.Beep(500, 50)
    ws.Beep(1000, 1000)


import pyautogui as pa
import time


def buy(sp, tp=False):
    pa.moveTo(1050, 280)
    time.sleep(1)
    pa.drag(0, 50)
    pa.leftClick()  # clicca su new trade
    time.sleep(1)
    if tp:
        pa.drag(0, 120)
        pa.leftClick()  # clicca tp
        time.sleep(1)
    else:
        pa.drag(0, 150)
        pa.leftClick()
        pa.doubleClick()
        pa.write('30')
        if sp > 0:
            pa.drag(0, 30)
            pa.leftClick()
            pa.doubleClick()
            pa.write(str(sp))
            pa.drag(0, -30)
        pa.drag(0, 100)
        pa.click()


def buy_tp(sp, tp, cap=25, mod=0):
    try:
        if mod == 0:
            posx = 1050, 280
        elif mod == 1:
            posx = 900, 180
        else:
            posx= 1050, 280
        pa.moveTo(posx)  # ARRIVA SUL PULSANTE DI BUY
        pa.leftClick()  # CLICCA SU BUY

        time.sleep(0.2)
        pa.drag(0, 95)  # SCENDE FINO ALLA CASELLA PER INSERIRE IL CAPITALE DA INVESTIRE
        time.sleep(0.2)
        pa.leftClick()  # CLICCA SULLA CASELLA
        time.sleep(0.2)
        pa.doubleClick()  # VA IN MODALITà SCRITTURA ED EVIDENZIA IL DEFAULT
        pa.write(str(cap))  # SCRIVE IL CAPITALE DA INVESTIRE

        time.sleep(0.1)
        pa.drag(0, 150 - 50)  # SCENDE NELLA SEZIONE DEL TP
        pa.leftClick()  # CLICCA SUL TP
        time.sleep(0.2)

        pa.doubleClick()
        pa.write(str(tp))  # SCRIVE IL TP

        pa.drag(0, 30)  # SCENDE ALLO STOP LOSS
        time.sleep(0.2)
        pa.leftClick()  # CLICCA SULLO STOP LOSS
        time.sleep(0.2)
        pa.doubleClick()  # EVIDENZIA
        pa.write(str(sp))  # SCRIVE LO STOP LOSS
        pa.drag(0, -30)
        pa.drag(0, 100)
        pa.click()  # CLICCA PER ATTIVARE L'AZIONE
    except Exception as e:
        print(e)



def sell_tp(sp, tp, cap=25, mod=0):
    try:
        if mod == 0:
            posx = 1050, 280
        elif mod == 1:
            posx = 900, 180
        else:
            posx= 1050, 280
        pa.moveTo(950, 280)  # ARRIVA SUL PULSANTE DI BUY
        pa.leftClick()  # CLICCA SU BUY
        # SCENDE FINO ALLA CASELLA PER INSERIRE IL CAPITALE DA INVESTIRE
        pa.drag(120, 105)  # CLICCA SULLA CASELLA
        pa.leftClick()  # VA IN MODALITà SCRITTURA ED EVIDENZIA IL DEFAULT
        time.sleep(0.2)  # SCRIVE IL CAPITALE DA INVESTIRE
        pa.doubleClick()  # SCENDE NELLA SEZIONE DEL TP
        pa.write(str(cap))  # CLICCA SUL TP
        # SCRIVE IL TP
        # SCENDE ALLO STOP LOSS
        time.sleep(0.1)  # CLICCA SULLO STOP LOSS
        pa.drag(0, 150 - 70)
        time.sleep(0.2)  # EVIDENZIA
        # SCRIVE LO STOP LOSS
        pa.leftClick()  # CLICCA PER ATTIVARE L'AZIONE
        time.sleep(0.1)
        pa.doubleClick()
        pa.write(str(tp))

        pa.drag(0, 30)
        time.sleep(0.2)
        pa.leftClick()
        time.sleep(0.2)
        pa.doubleClick()
        pa.write(str(sp))
        pa.drag(0, -30)
        pa.drag(0, 100)
        pa.click()
    except Exception as e:
        print(e)


def close():
    pa.moveTo(1420, 510)
    time.sleep(1)
    pa.drag(0, 50)
    pa.leftClick()  # clicca su new trade
    pa.drag(-500, -25)
    time.sleep(1)
    pa.leftClick()


def just_click():
    try:
        pa.moveTo(50, 250)
        pa.leftClick()  # clicca su new trade
    except Exception as e:
        print(e)
