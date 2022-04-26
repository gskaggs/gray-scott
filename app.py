import pandas as pd
import streamlit as st

st.title("Genetic Algorithms for the Physical Simulation of Flower Pigmentation")

st.markdown(
    '''
    An undergraduate thesis by _Grant Skaggs._

    Dank tunes:
    [https://open.spotify.com/track/6tuxI8MMx9T67pYjlMvgxG?si=90123bffd5d14d8a](https://open.spotify.com/track/6tuxI8MMx9T67pYjlMvgxG?si=90123bffd5d14d8a)

    ## Demo
    ''')

NUM_CHROMOSOMES = 20
NUM_SURVIVORS = 1

generation_id = st.session_state.get('generation_id', 1)
st.write(f'Current generation: {generation_id}')

options = list(range(NUM_CHROMOSOMES))
default_options = [] # options[:NUM_SURVIVORS]
prev_selections = st.session_state.get('selections', default_options)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

df = pd.DataFrame(list(range(20)))
csv = convert_df(df)

st.markdown("""
<style>
div.stButton > button:first-child {
    width: 100%;
}
div.stDownloadButton > button:first-child {
    width: 100%;
}
</style>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1: sim_button = st.button("Run Next Generation", disabled=len(prev_selections) != NUM_SURVIVORS)
with col2: reset_button = st.button("Reset")
with col3: download_button = st.download_button("Download Parameters", data=csv, file_name='chromosome_params.csv')

if sim_button:
    generation_id += 1
    st.session_state['generation_id'] = generation_id

if reset_button:
    pass

selections = st.multiselect(
     f'Select {NUM_SURVIVORS} Chromosomes',
     options,
     default_options,
     key=generation_id)

st.session_state['selections'] = selections

if len(prev_selections) != len(selections):
    st.experimental_rerun()

st.markdown(
    '''
    ## Motivation
    ## Background
    ## Methods
    ## Results
    ## Memes
    ''')

    