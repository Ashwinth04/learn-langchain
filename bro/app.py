from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
import streamlit as st

llm = Ollama(model="qwen2:1.5b")

text_generation_prompt = PromptTemplate(
    template="Generate some text about the following topic: {topic}",
    input_variables=["topic"]
)

summarization_prompt = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"]
)

qa_prompt = PromptTemplate(
    template="Answer the question based on the following text:\nText: {text}\nQuestion: {question}",
    input_variables=["text", "question"]
)

text_generation_chain = LLMChain(llm=llm, prompt=text_generation_prompt)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

st.title("Langchain App")

st.sidebar.title("Task Type")

task = st.sidebar.selectbox("Select a task", ["Text generation", "Text summarization", "Question Answering"])

if task == "Text generation":
    topic = st.text_input("Enter the topic name...")

    if topic:
        result = text_generation_chain.run(topic)
        st.write("Generated Text:", result)

elif task == "Text summarization":
    text_to_summarize = st.text_area("Enter the text that you want me to summarize...")

    if text_to_summarize:
        result = summarization_chain.run(text_to_summarize)
        st.write("Summary:", result)

elif task == "Question Answering":
    text = st.text_area("Enter the text...")
    question = st.text_input("Ask your question here...")

    if text and question:
        result = qa_chain.run({"text": text, "question": question})
        st.write("Answer:", result)
