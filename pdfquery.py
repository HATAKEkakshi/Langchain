import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from constants import openai_key
from typing_extensions import Concatenate

os.environ["OPENAI_API_KEY"]=openai_key

#provide the path of pdf file/files.
pdfreader=PdfReader('text.pdf')



#read text from pdf

raw_text =' '
for i ,page in enumerate(pdfreader.pages):
    content=page.extract_text()
    if content:
        raw_text+= content



#we need to split the text using character text soilt such that it should not increase token size 

text_spliter=CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function =len,

)

texts=text_spliter.split_text(raw_text) 
len(texts)

#dwonload emeddings from oepnai

embeddings=OpenAIEmbeddings()

document_search =FAISS.from_texts(texts,embeddings)



chain=load_qa_chain(OpenAI(),chain_type="stuff")

query="what is Rosa"
docs=document_search.similarity_search(query)
print(docs[0].page_content)
