import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools, Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor, create_react_agent
import streamlit as st 

#
load_dotenv()

#S
st.set_page_config("USE MIGRAPQ Agent for complexe tasks")
st.header("USE MIGRAPQ Agent for complexe tasks")
st.write("""
        MIGRAQ_AGENT is not a simple chat system, 
         but agent that can solve complexe problems. 
         It has the ability to search the internet, 
         solve math problems, as well as use other tools.
         """)
# Configuration du LLM
llm_groq = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

# Prompt ReAct (corrigé)
react_prompt = """
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt_template = PromptTemplate(
    template=react_prompt, 
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

# Définition des outils
# 1. Moteur de recherche
search_engine = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduckgo_search",
    description="Useful for searching the web for current information, facts, and data",
    func=search_engine.run,
)

# 2. Outil mathématique
math_tools = load_tools(["llm-math"], llm=llm_groq)

# Combiner tous les outils
tools = math_tools + [search_tool]

# Créer l'agent ReAct
react_agent = create_react_agent(llm_groq, tools, prompt_template)

# Créer l'executor
myagent = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # Limite pour éviter les boucles infinies
)

#
if input := st.chat_input("Input the task here"):
    st.chat_message("user").write(input)
    answer = myagent.invoke({"input": input})
    st.chat_message("assistant").write(answer["output"])

with st.sidebar:
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()