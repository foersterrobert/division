import streamlit as st
import pandas as pd
import numpy as np

st.title('Division Model')
st.write('Use a custom activation function to teach a small NN devision.')

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


