import os
import openai
import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone 
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
##lets read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
doc=read_doc('/Users/hemantkumar/Developer/Langchain/document/')
print(len(doc))
## divide the docs into chunks 

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)
print(documents)

##emdding technique of openai

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
print(embeddings)

vector=embeddings.embed_query("what is rosa")
print(vector)