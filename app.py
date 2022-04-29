import pandas as pd
import streamlit as st 
from PIL import Image as im
from sim_driver_gui import GuiSimulationDriver

st.title("Genetic Algorithms for the Physical Simulation of Flower Pigmentation")


st.markdown(
    '''
    An undergraduate thesis by _Grant Skaggs._

    ## Demo
    ''')

NUM_CHROMOSOMES = 100
NUM_SURVIVORS = 1

generation_id = st.session_state.get('generation_id', 1)
sim_driver = st.session_state.get('sim_driver', GuiSimulationDriver())

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
with col1: sim_button = st.button("Run Next Generation", disabled=len(prev_selections) < 1 and generation_id != 1)
with col2: reset_button = st.button("Reset")
with col3: download_button = st.download_button("Download Parameters", data=csv, file_name='chromosome_params.csv')

if sim_button:
    if generation_id != 1:
        sim_driver.register_preferred(prev_selections)
    generation_id += 1
    st.session_state['generation_id'] = generation_id
    st.session_state['cur_image'] = im.new("RGB", (2000, 1440), (0, 0, 255))

current_image = st.session_state.get('cur_image', im.new("RGB", (2000, 1440)))
st.image(current_image)

if reset_button:
    generation_id = 1
    st.session_state['generation_id'] = generation_id
    st.session_state['cur_image'] = im.new("RGB", (2000, 1440))
    st.session_state['sim_driver'] = GuiSimulationDriver()
    st.experimental_rerun()

selections = st.multiselect(
     f'Select at least 1 chromosome',
     options,
     default_options,
     key=generation_id)

st.session_state['selections'] = selections

if sim_button:    
    st.session_state['cur_image'] = sim_driver.run_generation(generation_id)
    st.experimental_rerun()

if len(prev_selections) != len(selections):
    st.experimental_rerun()