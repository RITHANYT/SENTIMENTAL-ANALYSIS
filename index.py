import streamlit as st
import streamlit.components.v1 as components
from nbconvert import PythonExporter
import nbformat
from IPython.display import display
import subprocess
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from R_cuts import positive_percentage
from R_cuts import negative_percentage
import streamlit as st
gif_url = "https://i.gifer.com/origin/7a/7a37c858c7d75aa0235d34e1c73694d0_w200.gif"

st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="{gif_url}" width="200"/></div>',
    unsafe_allow_html=True
)
st.title("REVIEW CUTS WEBSITE WELCOMES     YOU")
 
st.write("HERE ARE THE REVIWS FOR THE DATASET YOU CHOOSE")

css = """
<style>
    body {
        background-color: #000000;
        padding: 20px;
    }

    .circular-progress-bar {
        position: relative;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background-color: #ddd;
        overflow: hidden;
    }

    .circular-progress-bar .circle {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        clip: rect(0, 200px, 200px, 100px);
    }

    .circular-progress-bar .circle-fill-positive {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        clip: rect(0, 100px, 200px, 0);
        background-color: #2ecc71;
    }

    .circular-progress-bar .circle-fill-negative {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        clip: rect(0, 200px, 200px, 100px);
        background-color: #e74c3c;
    }

    .text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 24px;
        text-align: center;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.write(positive_percentage)
st.markdown('<div class="circular-progress-bar"><div class="circle"></div><div class="circle-fill-positive"></div></div>', unsafe_allow_html=True)
st.markdown(f'<p class="text">Good Reviews: {positive_percentage}%</p>', unsafe_allow_html=True)

st.write(negative_percentage)
st.markdown('<div class="circular-progress-bar"><div class="circle"></div><div class="circle-fill-negative"></div></div>', unsafe_allow_html=True)
st.markdown(f'<p class="text">Bad Reviews: {negative_percentage}%</p>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
        body {
            background-color: red;
        }
    </style>
    """,
    unsafe_allow_html=True
)
