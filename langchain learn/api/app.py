from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import  add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

app = FastAPI(
    title = "Langchain server",
    version = "1.0",
    description = "A langchain server for learning purpose"
)

llm = Ollama(model = "qwen2:1.5b")
prompt = ChatPromptTemplate.from_template("Wrtie me an essay about {topic} for 100 words")

add_routes(
    app,
    prompt|llm,
    path="/essay"
)

if __name__ == "__main__":
    uvicorn.run(app,host = "localhost", port = 8000)