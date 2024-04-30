from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
