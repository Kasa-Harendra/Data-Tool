import streamlit as st
import pandas as pd
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

def upload_file() -> pd.DataFrame | None:
    df = None
    file = st.file_uploader("Upload your file: ")
    if file:
        try:
            df = read_file(file)
            if 'file_name' not in st.session_state:
                st.session_state['file_name'] = file.name
                st.session_state['editable_df_filename'] = file.name
            elif st.session_state.file_name != file.name:
                st.session_state.file_name = file.name
                 
        except ValueError as e:
            st.error(f"{e}")
    else:
        st.subheader("Please upload a file")
    return df

def read_file(file:UploadedFile | None) -> pd.DataFrame : 
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('json'):
        df = pd.read_json(file)
    else:
        raise ValueError("Invalid File Type: Please enter a valid file(csv, excel, json)")  
    if df.empty:
        raise ValueError("Empty file")
    return df
        
        
def show_all_data_details( df:pd.DataFrame | None , tab:DeltaGenerator) -> None:
    if df is None:
        return
    data_container = tab.container(border=True)
    with data_container:
        st.text("Data Frame: ")
        st.dataframe(data=df)
        st.markdown(f"DataFrame shape: **{df.shape}**")
    
    statistics_container = tab.container(border=True)
    with statistics_container:
        st.text("Statistics: ")
        st.dataframe(data=df.describe(), use_container_width=True)

    info = {
        "Columns" : df.columns.to_list(),
        "Null Count": df.isnull().sum().to_list(),
        "Data Types": df.dtypes.to_list(),
        "Null Percent": (df.isnull().sum()/df.shape[0])*100
    }
    info_container = tab.container(border=True)
    with info_container:
        st.text("Columns Info")
        st.dataframe(data=info, use_container_width=True, hide_index=True)
        null_percent = (df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100 
        st.markdown(f"Null Value Percentage(Across Dataset): **{null_percent}**")