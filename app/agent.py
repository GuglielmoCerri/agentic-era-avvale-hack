import json
import matplotlib.pyplot as plt
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode
from google.cloud import bigquery
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

import json
import matplotlib.pyplot as plt

# Initial configurations
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# Initialize BigQuery client
client = bigquery.Client()

# Set up language model (the same LLM will be used for agents)
llm = ChatVertexAI(
    model=LLM,
    location=LOCATION,
    temperature=0,
    max_tokens=1024,
    streaming=True, 
)

# ------------- TOOLS DEFINITIONS ---------------
@tool
def query_bigquery(query_str: str) -> list[dict]:
    """
    Executes a SQL query on BigQuery.
    The query_str parameter must be a JSON string in the format:
      {"query": "SELECT * FROM ..."}
    Returns the query results as a list of dictionaries.
    """
    try:
        data = json.loads(query_str)
        sql_query = data["query"]
        print(f"Executing query: {sql_query}")
    except (json.JSONDecodeError, KeyError) as e:
        return [{"error": f"Invalid input format: {e}"}]
   
    try:
        query_job = client.query(sql_query)
        results = query_job.result()
        rows = [dict(row) for row in results]
        return rows
    except Exception as e:
        return [{"error": f"Error executing query: {str(e)}"}]

@tool
def generate_graph(json_input: str) -> str:
    """
    Generates a graph using matplotlib in a generalized way.
    
    The json_input parameter must be a JSON string with the following format:
    
    {
        "chart": "bar",           # Chart type: 'bar', 'line', 'pie', 'scatter'
        "data": [
            {"label": "A", "value": 10},
            {"label": "B", "value": 20}
        ],
        "title": "Chart Title",      # Optional
        "xlabel": "X Label",          # Optional
        "ylabel": "Y Label",          # Optional
        "figsize": [8, 6]             # Optional: figure size (width, height)
    }
    
    Saves the chart as 'graph.png' and returns a confirmation message.
    """
    try:
        config = json.loads(json_input)
        chart_type = config.get("chart")
        data_points = config.get("data")
        title = config.get("title", "Generated Chart")
        xlabel = config.get("xlabel", "X")
        ylabel = config.get("ylabel", "Y")
        figsize = config.get("figsize", [8, 6])
    except Exception as e:
        return f"Error parsing JSON: {str(e)}"
    
    # Validate required data
    if not chart_type or not data_points:
        return "Error: 'chart' and 'data' fields are required."
    
    if not isinstance(data_points, list) or not all(
        isinstance(dp, dict) and "label" in dp and "value" in dp for dp in data_points
    ):
        return "Error: 'data' field must be a list of objects with 'label' and 'value' keys."
    
    labels = [dp["label"] for dp in data_points]
    values = [dp["value"] for dp in data_points]
    
    plt.figure(figsize=figsize)
    
    if chart_type == "bar":
        plt.bar(labels, values)
    elif chart_type == "line":
        plt.plot(labels, values, marker='o')
    elif chart_type == "pie":
        plt.pie(values, labels=labels, autopct='%1.1f%%')
    elif chart_type == "scatter":
        # For scatter plot we use indices as x coordinates
        plt.scatter(range(len(values)), values)
        plt.xticks(range(len(labels)), labels)
    else:
        return "Error: Unsupported chart type. Use 'bar', 'line', 'pie', or 'scatter'."
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()
    
    return "Chart generated and saved as 'graph.png'."

# -------- AGENT NETWORK DEFINITION --------

members = ["conversation assistant", "graph creator", "data analyst"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

# Modified orchestrator system prompt to ensure proper routing
orchestrator_system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next."
    "\n\nIMPORTANT ROUTING RULES:"
    "\n1. If the user message is a greeting (like 'Hello', 'Hi', etc.) or a general question,"
    " route to 'conversation assistant'."
    "\n2. If the user asks for data analysis or information that requires database queries"
    " (like 'give me the top 5 best-selling products'), route to 'data analyst'."
    "\n3. If the user asks for visualizations or charts based on data, route to 'graph creator'."
    "\n4. Route to FINISH only when an agent has provided a complete response to the user's query."
    "\n5. ALWAYS look at the latest user message to determine the appropriate route."
)


class TaskAssignment(BaseModel):
    """Schema for assigning a task to an agent."""
    next: Literal[*options] = Field(..., description="The task to be assigned to an agent.")


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]


class State(MessagesState):
    next: str


# Simplified orchestrator node without the conversation_handled flag
def orchestrator_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = state["messages"]
    
    # Controlla se l'ultimo messaggio proviene da un agente
    if len(messages) > 0 and hasattr(messages[-1], 'name') and messages[-1].name in members:
        # Se l'ultimo messaggio è da un agente, attendi un nuovo input dall'utente
        return Command(goto=END, update={"next": END})
    
    # Prepara un unico messaggio per l'orchestratore che include il sistema e l'input dell'utente
    prompt = f"{orchestrator_system_prompt}\n\nUser says: {messages[-1].content}\n\nWhich worker should handle this request?"
    
    # Usa un prompt diretto invece di gestire complesse strutture di messaggi
    response = llm.invoke(prompt)
    
    # Analizza la risposta per determinare il routing
    content = response.content.lower()
    
    if "data analyst" in content:
        goto = "data analyst"
    elif "graph creator" in content:
        goto = "graph creator"
    elif "conversation assistant" in content:
        goto = "conversation assistant"
    else:
        goto = END
    
    return Command(goto=goto, update={"next": goto})


# Conversation assistant node
def conversation_assistant_node(state: State) -> Command[Literal["orchestrator"]]:
    # Get messages
    messages = state["messages"]
    
    # Prepare system prompt for conversation assistant
    conversation_system_prompt = (
        "You are a friendly, conversational assistant. "
        "Respond to greetings and general questions in a colloquial manner. "
        "When introducing yourself, briefly explain that you can: "
        "1) Analyze data from the BigQuery database with the sales_online table "
        "2) Create charts to visualize data "
        "Don't offer to run queries or create charts unless specifically requested by the user."
        "\n\nIMPORTANT: Provide a BRIEF and CONCISE response. "
        "Don't repeat the same information if the user sends multiple greetings. "
        "If the user is just greeting, respond with a simple greeting and a brief description of your capabilities."
    )
    
    # Add system prompt to the beginning of messages
    system_message = {"role": "system", "content": conversation_system_prompt}
    all_messages = [system_message] + messages
    
    # Get response from the model
    response = llm.invoke(all_messages)
    
    # Create a response message with the agent's name
    return Command(
        update={
            "messages": [
                HumanMessage(content=response.content, name="conversation assistant")
            ]
        },
        goto="orchestrator",
    )


graph_creator_agent = create_react_agent(
    llm, 
    tools=[generate_graph], 
    prompt="You are a helpful AI assistant who must provide a JSON command to generate a graph using matplotlib. "
        "Return a JSON in the following format: {\"chart\": \"bar\", \"data\": [{\"label\": \"A\", \"value\": 10}, ...]} "
        "Plot to the screen the image"
        "Optionally, you can include parameters such as 'title', 'xlabel', 'ylabel', and 'figsize' to customize the graph appearance."
)

def graph_creator_node(state: State) -> Command[Literal["orchestrator"]]:
    result = graph_creator_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="graph creator")
            ]
        },
        goto="orchestrator",
    )


data_analyst_agent = create_react_agent(
    llm, 
    tools=[query_bigquery], 
    prompt = "You are a helpful AI assistant who must provide a SQL BigQuery query "
        "to analyze data as the user requests. You can query data from 2 tables using BigQuery in `qwiklabs-gcp-03-d68fba73ee2d.sales`. "
        "Table 1 (sales_online): "
        "Order_ID (INTEGER), Date (DATE), Customer (STRING), Product (STRING), "
        "Quantity (INTEGER), Price_per_unit (INTEGER), Total_Amount (INTEGER), "
        "Payment_Method (STRING). \n"
        "Table 2 (product): "
        "ProductID (INTEGER), ProductName (STRING), Category (STRING), Price (INTEGER)"
        "You can join the 2 tables to extract relevant informations"
        )

def data_analyst_node(state: State) -> Command[Literal["orchestrator"]]:
    result = data_analyst_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="data analyst")
            ]
        },
        goto="orchestrator",
    )


# Define the workflow
workflow = StateGraph(State)
workflow.add_edge(START, "orchestrator")
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("conversation assistant", conversation_assistant_node)
workflow.add_node("graph creator", graph_creator_node)
workflow.add_node("data analyst", data_analyst_node)

# Compile the agent
agent = workflow.compile()

# Get the image in PNG format
png_data = agent.get_graph().draw_mermaid_png()

# Save the image to a file named "graph.png"
with open("graph_flow.png", "wb") as file:
    file.write(png_data)

#print("The graph has been saved as 'graph.png'")