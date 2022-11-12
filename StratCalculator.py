#import ADAM
import streamlit as st

""""""

st.title("Perla Enviroment Strat. Stat. Calculator")

stocks = ('Bottabile', 'Non Bottabile')
selected_stock = st.selectbox("Bottabile?", stocks)

n_days = st.slider("giorni testati", 1, 700)
mesi_testati = n_days/30

st.subheader("Dati Strategia")
data = st.text_input('Nome Strategia ')
asset_timeframe = 'nan'
spread = float(st.text_input('Spread Medio Asset '))
leva = int(st.text_input('leva '))
tp = float(st.text_input('Take Profit = '))

sl = float(st.text_input('Stop Loss = '))
wins = int(st.slider("Wins con Take Profit", 1, 25))
winss = str(st.text_input("Wins (senza take profit es: 0.98 0.50 045 with SPACES) usa 'nan' se non ne hai  = "))
losses = int(st.slider("Losses con stop loss", 1, 25))
lossess = str(st.text_input("Losses (solo senza stop loss = 0.98 0.50 045 with SPACES usa 'nan' se non ne hai =  "))
mesi = mesi_testati
index = 0
if False:
    data = str(input('Nome Strategia '))
    asset_timeframe = str(input('Asset e Timeframe esatti per YFINANCE con spazio (se non si vuole usare inserire nan)'))
    spread = float(input('Spread Medio Asset '))
    leva = int(input('leva '))
    tp = float(input('Take Profit = '))
    sl = float(input('Stop Loss = '))
    wins = int(input("Wins (con take profit) = "))
    winss = str(input("Wins (senza take profit es: 0.98 0.50 045 with SPACES) usa 'nan' se non ne hai  = "))
    losses = int(input("Losses (solo con stop loss) = "))
    lossess = str(input("Losses (solo senza stop loss = 0.98 0.50 045 with SPACES usa 'nan' se non ne hai =  "))
    mesi = float(input('mesi testati = '))
    # format = str(input('basic, rob = '))
    index = 0


def varianciator(a_tf):
    a_tf.split(" ")
    asset = a_tf[0]
    timeframe = a_tf[1]
    print(a_tf)
    print(timeframe)
    df = ADAM.incredible_data_requestor_d(asset, timeframe, 1)
    var = ADAM.variance(df)
    return var

if asset_timeframe != 'nan':
    var = varianciator(asset_timeframe)



else:
    var = 'NaN'
if winss == 'nan':
    winssf = 0
    winss_profit = 0
    winss = 0
else:
    try:
        winssf = winss.split(' ')
        for i in range(len(winssf)):
            winssf[i] = float(winssf[i])
        winss_profit = sum(winssf)
        winss = len(winssf)
    except Exception as e:
        print(e)
        winssf = float(winss)
        winss_profit = winssf
        winss = 1

if lossess == 'nan':
    lossessf = 0
    lossess_profit = 0
    lossess = 0
else:
    try:
        lossessf = lossess.split(' ')
        for i in range(len(lossessf)):
            lossessf[i] = float(lossessf[i])
        lossess_profit = sum(lossessf)
        lossess = len(lossessf)
    except Exception as e:
        print(e)
        lossessf = float(winss)
        lossess_profit = lossessf
        lossess = 1

format = 'rob'

# print(winss, len(winssf))

all_wins = wins + winss
all_losses = losses + lossess
all_tested = all_wins + all_losses
perc_wins = wins*tp + winss_profit
perc_losses = losses*sl + lossess_profit
profit = perc_wins - perc_losses
profit_leved_spreaded = (profit - (spread * all_tested)) * leva


formato = f"***TITOLO*** »({data})(***Grado*** II)\nDurata Strategia ***»*** {mesi} mesi\nDurata Strategia ***»*** {mesi * 30.25} giorni\n{round(((wins + losses + winss + lossess) / (mesi * 30.25)), 3)} operazioni al giorno\n{round(((wins + losses + winss + lossess) / (mesi * 30.25) * 7), 3)} operazioni a settimana\nBottabile ***»*** {selected_stock}\nStop Loss ***»*** {sl}\nTake Profit ***»*** {tp}\nPosizioni Testate ***»*** = {all_tested}\nOperazioni Vinte ***»*** {all_wins} ({round((all_wins / all_tested * 100), 3)}%)\nOperazioni Perse ***»*** {all_losses} ({round(100 - (all_wins / all_tested * 100), 3)}%)\nProfitto Percentuale ***»*** {round(profit, 3)}%\nProfitto Percentuale (con leva e spread) ***»*** {round(profit_leved_spreaded, 3)}\nGuadagno ***»*** {round(perc_wins, 3)}%\nGuadagno medio ***»*** {round(perc_wins / all_wins, 3)}%\nPerdite ***»*** {round(perc_losses, 3)}%\nVarianza Asset ***»*** {var}\nPerdite medie ***»*** {round((perc_losses / all_losses), 3)}%\nPerdite spread ***»*** {round((spread * all_tested), 3)}%\n***GUADAGNO MENSILE*** ***»*** {round(profit_leved_spreaded / mesi, 3)}%"


print(formato)
#Pier_Gitto1
st.write(formato)

