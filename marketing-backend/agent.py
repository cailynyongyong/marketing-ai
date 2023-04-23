from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import openai
import os
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

st.title("Marketing Intern")
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

def get_text():
    input_text = st.text_input("Type in the product website below", key="input")
    return input_text 

user_input = get_text()

if(user_input):
    loader = WebBaseLoader(user_input)
    # "https://blog.celestia.org/introducing-rollkit-a-modular-rollup-framework/"
    docs = loader.load()
    texts = text_splitter.split_documents(docs)
    db = FAISS.from_documents(texts, embeddings)
    product = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

    tools = [
        Tool(
        name = "Information",
        func=product.run,
        description="useful for when you need to get information about the product."
        ),
    ]   

    # Set up the base template
    template = """Create an advertising post for a product that the user inputs. You have access to the following toolkits:

    {tools}

    Use the following format:

    Product: the input product you must advertise
    Thought: think about creating an advertising post 
    Action: the action to take, ask questions using [{tools}] to get relevant product information
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember advertise the product accustomed to Twitter. Be concise and use hashtags.

    Product: {input}
    {agent_scratchpad}"""

    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
        
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    class CustomOutputParser(AgentOutputParser):
        
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            
            action_input = match.group(1)
            # Return the action and action input
            return AgentAction(tool="Information", tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation: "], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

if(user_input):
    response = agent_executor.run("Create a marketing post about Rollkit.")
    st.write('**Result:** \n')
    st.write(response)


