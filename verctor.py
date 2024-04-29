import os
from langchain_core.runnables.base import chain
import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA 
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import pinecone
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(   
    model_name='gpt-3.5-turbo',  
    temperature=0.7
) 
chain=load_qa_chain(llm=llm,chain_type="stuff")
##lets read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
doc=read_doc('/Users/hemantkumar/Developer/Langchain/document/')
len(doc)
## divide the docs into chunks 

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)

##emdding technique of openai

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

vector=embeddings.embed_query("what is rosa")
len(vector)


##vector serach in pinecone 
index_name="langchainvector"


index = PineconeVectorStore.from_documents(
        doc,
        index_name=index_name,
        embedding=embeddings
    )
#cosine similarity retreive results
def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results
qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=index.as_retriever()
) 
#search answer from vector database
def retireve_answer(query):
    doc_search=retrieve_query(query)
    response=chain.run(input_documents=doc_search,question=query)
    return response

our_query=input("Ask your query")
answer=retireve_answer(our_query)
print(answer)


"""def retireve_answer(query):
    doc_search=retrieve_query(our_query)
    print(doc_search)
    response=chain.invoke({"input":"doc_search","question":our_query})
    return response
answer=retireve_answer(our_query)
print(answer)"""