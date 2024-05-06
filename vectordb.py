from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import faiss
from dotenv import load_dotenv
import streamlit as st



load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# Reading the text from pdf page by page and storing it into various




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



#Getting the text into number of chunks as it is helpful in faster processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Storing the text chunks into embeddings to retrive the answer for the query outoff it
def get_vector_store(text_chunks, file_type, filename):
    name = filename.split(".")[0]
    vector_store = faiss.FAISS.from_texts(text_chunks, embedding=embeddings)
    file_type = file_type
    vector_store.save_local(f"faiss_index/{file_type[0]}", index_name=name)
    
  
def get_query_data(question, index, directory):
    db = faiss.FAISS.load_local(f"faiss_index/{directory}", embeddings, allow_dangerous_deserialization=True, index_name=index)
    retriever = db.as_retriever(k=4)
    docs_data = retriever.invoke(question)
    return docs_data
 
def append_to_store(text_chunks,directory):
    index_path = f'faiss_index/{directory[0]}'
    if os.path.exists(index_path+'/index.faiss'):
        vector_store = faiss.FAISS.load_local(f'faiss_index/{directory[0]}', embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(text_chunks)
        vector_store.save_local(f"faiss_index/{directory[0]}")
    else:
        vector_store = faiss.FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(f"faiss_index/{directory[0]}", index_name="index")
    

# def append_to_store(text_chunks, directory):
#     index_path = f'faiss_index/{directory[0]}'
#     if os.path.exists(index_path):
#         vector_store = faiss.FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True, index_name="index")
#     else:
#         vector_store = faiss.FAISS(embeddings)
        
#     vector_store.add_texts(text_chunks)
#     vector_store.save_local(index_path, index_name="index")
