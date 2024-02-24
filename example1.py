##integeration with openai
import os
import openai
import langchain_community
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate

import streamlit as st
os.environ["OPENAI_API_KEY"]=openai_key
#stream lit framework

st.title("Langchain demo with openai api")
input_text=st.text_input("search the topic u want")
#openai llms
llm=OpenAI(temperature=0.8)
if input_text:
    st.write(llm(input_text))