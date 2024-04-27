import os
from langchain_openai import OpenAI
from dotenv import load_dotenv # type: ignore
load_dotenv()
import streamlit as st

#function to load openai model and get response 
def get_openai_response(question):
    llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.7)
    response=llm(question)
    return response
##initialize our streamlit app

st.set_page_config(page_title="Q$A Demo")

st.header("Langchain Application")
input=st.text_input("Input : ",key="input")
response=get_openai_response(input)
submit=st.button("Ask the question")

##if submit button is clicked

if submit:
    st.subheader("The Response is ")
    st.write(response) # type: ignore
