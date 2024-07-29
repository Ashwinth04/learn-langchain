import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import time

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]


if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.db = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Groq demo")
llm = ChatGroq(groq_api_key = groq_api_key,model="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template("""
Answer the following questions based only one the given context. Dont use profane language and make sure that you provide the most accurate answer.<context> {context} </context>. Question: {input}
""")

document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
retriever = st.session_state.db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Enter your question here...")

if prompt:
    start = time.time()
    response = retrieval_chain.invoke({"input":prompt})
    print(f"Response time: {time.time() - start}")
    st.write(response['answer'])