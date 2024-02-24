##integeration with openai
import os
import openai
import langchain_community
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory


import streamlit as st
os.environ["OPENAI_API_KEY"]=openai_key
#stream lit framework

st.title("Celeberity search bar")
input_text=st.text_input("search the topic u want")
#Memory
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
description_memory = ConversationBufferMemory(input_key='dob',memory_key='description_memory')
#Prompt tempelate
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celeberity{name}"
)
#openai llms
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born "
    )
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)
third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Metion 5 major event happen on {dob} in the world"
    )
chain3=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='description',memory=description_memory)
parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Person name'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(description_memory.buffer)