import streamlit as st
import subprocess

st.title("Document Processing Tool")

option = st.selectbox("Choose the processing type:", ("select type","Table Extraction", "Text Extraction", "Graph Extraction"))

if option == "Table Extraction":
    st.write("You selected Table Extraction.")
    st.write("Running Table Extraction script...")
    subprocess.run(["streamlit", "run", "tablepdf.py"])
elif option == "Text Extraction":
    st.write("You selected Text Extraction.")
    st.write("Running Text Extraction script...")
    subprocess.run(["streamlit", "run", "app3.py"])
elif option == "Graph Extraction":
    st.write("You selected Graph Extraction.")
    st.write("Running Graph Extraction script...")
    subprocess.run(["streamlit", "run", "StreamlitGraph.py"])
