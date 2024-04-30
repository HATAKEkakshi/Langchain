from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
pdfreader=PdfReader("/Users/hemantkumar/Developer/Langchain/document/61536ffd-be39-4921-9c2d-471940e7460d_Hackathon_Idea.pdf")
add opeai and astra db token and astra db  id
#read text from pdf
raw_text=" "
for i, page in enumerate(pdfreader.pages):
        content=page.extract_text()
        if content:
                raw_text +=content
#initzlize the connection to your database

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)
llm=OpenAI(openai_api_key=OPENAI_API_KEY)
embeddings=OpenAIEmbeddings(openai_api_type=OPENAI_API_KEY)

astra_vector_store=Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None,
)
text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
)
texts=text_splitter.split_text(raw_text)
#load the dataset into vectorstore
astra_vector_store.add_texts(texts[:50])
print("Insterted %i headlines."%len(texts[:50]))
astra_vector_index=VectorStoreIndexWrapper(vectorstore=astra_vector_store)
first_question=True
while True:
        if first_question:
                query_text=input("\n Enter your question : ").strip()
        else:
                 query_text=input("\n What is your next question: ").strip()
        if query_text.lower() == "quit":
                break
        if query_text == "":
                continue
        first_question=False
        print("\n Question : \"%s\"" % query_text)
        answer=astra_vector_index.query(query_text,llm=llm).strip()
        print("ASNWER:\"%s\"\n" % answer)
        print("FIRST DOCUMENT BY RELEVANCE:")
        for doc,score in astra_vector_store.similarity_search_with_score(query_text, k=4):
                print("        [%0.4f] \"%s ...\"" % (score,doc.page_content[:84]))