import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from dotenv import load_dotenv # type: ignore
load_dotenv()


os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

#function to load openai model and get response 
def get_gemini_response(question):
   llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
   response=llm(question)
   return response
##initialize our streamlit app

st.set_page_config(page_title="Q$A Demo")

st.header("infobit.py") 
input=st.text_input("Input : ",key="input")
response=get_gemini_response(input)
submit=st.button("Ask the question")

##if submit button is clicked

if submit:
    st.subheader("The Response is ")
    st.write(response) # type: ignore
