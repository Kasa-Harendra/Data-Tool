import streamlit as st
from utilities import *
from data import *
from edit_df import *
from sidetool import *

#Page Configurations
st.set_page_config(page_title="Data Tool", page_icon="📊")

#title
st.title("Data Tool")
st.divider()

sidebar_content()

#Upload file
data = upload_file()
st.divider()

if data is not None:
    st.session_state['main_df'] = data.copy()
    if 'editable_df' not in st.session_state:
        st.session_state['editable_df'] = st.session_state['main_df'].copy()
    elif st.session_state['editable_df_filename'] != st.session_state['file_name']:
        st.session_state['editable_df'] = st.session_state['main_df'].copy()
        st.session_state['editable_df_filename'] = st.session_state['file_name']

    #Show data and info
    tab1, tab2, tab3 = st.tabs(tabs=["Data", "Visualization", "Preprocessing"])
    tab1.header("Given Data ")
    show_all_data_details(st.session_state['main_df'], tab1)
    tab1.divider()

    #Visualizations
    tab2.header("Visualization")
    if 'visual_df' not in st.session_state:
        st.session_state['visual_df'] = st.session_state['main_df'].copy()
    visualize_choice(tab2)
    tab2.divider()
    
    #Edit data
    tab3.header("Data Editor")
    editable_panel(tab3)
    