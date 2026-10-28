"""Streamlit dashboard for Steering Studio. Run: streamlit run dashboard.py"""
import streamlit as st
from activation_steering import run
st.set_page_config(page_title="Steering Studio",layout="wide")
st.title("Steering Studio")
st.warning("Experimental activation intervention. Compare controls and regressions; do not infer a reliable semantic control from one prompt pair.")
model=st.text_input("GPT-2-style model ID","gpt2")
positive=st.text_area("Positive direction prompt","I feel confident, clear, and decisive.")
contrast=st.text_area("Contrast direction prompt","I feel uncertain, confused, and hesitant.")
prompt=st.text_input("Generation prompt","I feel")
c1,c2=st.columns(2)
with c1: layer=st.number_input("Transformer block",min_value=0,max_value=47,value=6)
with c2: scale=st.slider("Steering scale",min_value=-2.0,max_value=2.0,value=0.3,step=0.1)
device=st.selectbox("Device",["auto","cuda","mps","cpu"])
if st.button("Compare baseline and intervention",type="primary"):
    with st.spinner("Running baseline and intervention…"):
        result=run(model,positive,contrast,prompt,int(layer),float(scale),device=device)
    a,b=st.columns(2)
    with a: st.subheader("Baseline");st.write(result["baseline"])
    with b: st.subheader("Intervened");st.write(result["steered"])
    st.json({"layer":result["layer"],"scale":result["scale"],"device":result["device"]})
