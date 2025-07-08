import streamlit as st
from streamlit.delta_generator import DeltaGenerator

def sidebar_content():
    sb = st.sidebar
    sb.subheader("Calculator")
    expression = sb.text_input(label="Enter the expression", placeholder='Ex: 2 + 3 * (4 / 6)')
    result = sb.container(border=True)
    with result:
        try:
            answer = eval(expression)
            st.markdown(f"**Answer**:    _{answer}_")
        except Exception as e:
            print("ERROR")