from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import openai
import os

# Load environment variables from the .env file
load_dotenv()

openai.api_key= os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0)

# from pathlib import Path
# relevant_parts = []
# for p in Path(".").absolute().parts:
#     relevant_parts.append(p)
#     if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
#         break
# doc_path = str(Path(*relevant_parts) / "processed" / "scraped.csv")
# print(doc_path)

# from langchain.document_loaders.csv_loader import CSVLoader
# loader = CSVLoader(doc_path)
# documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
# docsearch = FAISS.from_documents(texts, embeddings)

# def retrieval_qa_chain(query):
#     celestia = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k":3}), return_source_documents=True)
#     response = celestia({"question": query})
#     return response["answer"]

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://blog.celestia.org/introducing-rollkit-a-modular-rollup-framework/")
docs = loader.load()
celestia_texts = text_splitter.split_documents(docs)
celestia_db = FAISS.from_documents(celestia_texts, embeddings)
celestia = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=celestia_db.as_retriever())



# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

tools = [
    Tool(
        name = "Celestia QA System",
        func=celestia.run,
        description="useful for when you need to answer questions about Celesita's products. Input should be a fully formed question.",
        return_direct=True
    ),
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is Rollkit?")