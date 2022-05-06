import os
from email.policy import default
from statistics import median
import pandas as pd
import streamlit as st 
from PIL import Image as im
from sim_driver_gui import GuiSimulationDriver
import numpy as np

print(os. getcwd())
st.set_page_config('GA for Reaction Diffusion', '🌸')

markdown_file = open('./README.md', 'r')
markdown = markdown_file.read()
markdown = markdown[:markdown.rfind('\n')]
markdown = markdown[:markdown.rfind('\n')]
st.markdown(markdown)

default_pop_size = 20
pop_size = st.session_state.get('pop_size', default_pop_size)

generation_id = st.session_state.get('generation_id', 1)
sim_driver = st.session_state.get('sim_driver', GuiSimulationDriver())

st.write(f'Current generation: {generation_id}')

options = list(range(1, pop_size+1))
default_options = []
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
with col1: dt_slider   = st.slider('dt', 0.001, 1.0, .5)
with col2: T_slider    = st.slider('T',  0, 5000, 10, 10)
with col3: pop_slider  = st.slider('Population size', 1, 200, 20, disabled=generation_id>1)
    
pop_size = pop_slider
st.session_state['pop_size'] = pop_size

st.write('<style>div.row-widget.stRadio > div{flex-direction:row; }</style>', unsafe_allow_html=True)
model_radio = st.radio('Reaction-Diffusion Model', ['Gray-Scott___', 'Gierer-Meinhardt___', 'GenRD'])
fitness_radio = st.radio('Fitness Type', ['Dirichlet___', 'User-Input'])

col1, col2, col3 = st.columns(3)
with col1: sim_button = st.button("Run Next Generation", disabled=(len(prev_selections) < 1 and generation_id != 1 and fitness_radio=='User-Input'))
with col2: reset_button = st.button("Reset")
with col3: download_button = st.download_button("Download Parameters", data=csv, file_name='chromosome_params.csv')

if sim_button:
    if generation_id != 1:
        sim_driver.register_preferred(prev_selections)
    generation_id += 1
    st.session_state['generation_id'] = generation_id
    memes = os.listdir('thesis_memes')
    meme_id  = st.session_state.get('meme_id', 0)
    st.session_state['meme_id'] = (1 + meme_id) % len(memes)
    meme  = im.open('thesis_memes/'+memes[meme_id])
    w, h = meme.size
    st.session_state['cur_image'] = meme.resize((2000, int(h * 2000 / w)))

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
    key=generation_id,
    disabled=fitness_radio!='User-Input')

st.session_state['selections'] = selections

sim_driver.set_params(dt_slider, T_slider, model_radio, fitness_radio, pop_size)

if sim_button:    
    st.session_state['cur_image'] = sim_driver.run_generation(generation_id)
    st.experimental_rerun()

if len(prev_selections) != len(selections):
    st.experimental_rerun()

