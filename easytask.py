""" Easy task """
import streamlit as st

t = 1.37
s = 0.26
i = int(st.text_input('Soldi da investire'))

l = (t - 0.02) * 30
m = (s + 0.02) * 30
x = l * i / 100
y = m * i / 100
st.write("il take profit è di", x, "$")
st.write("il Stop Loss è di", y, "$")
