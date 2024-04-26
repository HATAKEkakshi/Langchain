##integrate our code with Openai api
import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
os.environ["OPENAI_API_KEY"]=openai_key

##streamlit framework
st.title('Celeberaity serach Results')
input_text=st.text_input("Search the topic u want")

#Prompt Template

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celeberity  {name}"

)

#memory
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
desc_memory=ConversationBufferMemory(input_key='dob',memory_key='desc_history')
##openai llms
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

#Prompt Template

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="When was {person} borm"

)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)


#Prompt Template

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob}"

)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='desc',memory=desc_memory)
parent_chain=SimpleSequentialChain(chains=[chain,chain2],verbose=True)
parent_chain1=SequentialChain(chains=[chain,chain2],input_variables=['name'],output_variables=['person','dob'],verbose=True)
parent_chain2=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','desc'],verbose=True)
if input_text:
    st.write(parent_chain2({'name':input_text}))
    
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    
    with st.expander('DOB Events'):
        st.info(dob_memory.buffer)

    with st.expander('Major Events'):
        st.info(desc_memory.buffer)