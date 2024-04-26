import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import faiss
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

print(raw_text)