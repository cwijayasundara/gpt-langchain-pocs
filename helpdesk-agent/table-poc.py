import streamlit as st
import pandas as pd

# Load some example data.
DATA_URL = \
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
data = st.cache(pd.read_csv)(DATA_URL, nrows=1000)

# Select some rows using st.multiselect. This will break down when you have >1000 rows.
st.write('### Full Dataset', data)
selected_indices = st.multiselect('Select rows:', data.index)
selected_rows = data.loc[selected_indices]
st.write('### Selected Rows', selected_rows)