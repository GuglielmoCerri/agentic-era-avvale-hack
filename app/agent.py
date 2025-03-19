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

# Configurazioni iniziali
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# Inizializza il client BigQuery
client = bigquery.Client()


# Imposta il modello di linguaggio (lo stesso LLM verrà usato per gli agenti)
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
    Esegue una query SQL su BigQuery.
    Il parametro query_str deve essere una stringa JSON nel formato:
      {"query": "SELECT * FROM ..."}
    Restituisce i risultati della query come lista di dizionari.
    """
    try:
        data = json.loads(query_str)
        sql_query = data["query"]
        print(f"Esecuzione query: {sql_query}")
    except (json.JSONDecodeError, KeyError) as e:
        return [{"error": f"Formato di input non valido: {e}"}]
   
    try:
        query_job = client.query(sql_query)
        results = query_job.result()
        rows = [dict(row) for row in results]
        return rows
    except Exception as e:
        return [{"error": f"Errore nell'esecuzione della query: {str(e)}"}]

@tool
def execute_query(state: MessagesState) -> dict:
    """
    Estrae ed esegue la query SQL dalla risposta del modello.
    Cerca un JSON formattato come {"query": "..."}.
    """
    last_message = state["messages"][-1]
    content = last_message.content
    json_query = None
    # Cerca il JSON racchiuso in ```json
    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        if json_end > json_start:
            json_query = content[json_start:json_end].strip()
    # Altrimenti cerca il JSON direttamente
    if not json_query and "{\"query\":" in content:
        json_start = content.find("{\"query\":")
        brace_count = 1
        for i in range(json_start + 1, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if brace_count == 0:
            json_query = content[json_start:json_end].strip()
    
    if json_query:
        result = query_bigquery(json_query)
        result_message = AIMessage(
            content=f"Ho eseguito la query e ecco i risultati:\n\n{result}"
        )
        return {"messages": [result_message]}
    
    error_message = AIMessage(
        content="Non sono riuscito a estrarre una query SQL valida dalla risposta."
    )
    return {"messages": [error_message]}

@tool
def execute_graph_from_response(state: MessagesState) -> dict:
    """
    Extract the JSON command to generate the graph from the model response 
    and execute it using the generate_graph tool.
    Looks for the JSON between ```json ... ``` or directly as a string.
    """
    print("graph creation")
    last_message = state["messages"][-1]
    content = last_message.content
    json_command = None

    # Cerca il blocco JSON delimitato da ```json ... ```
    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        if json_end > json_start:
            json_command = content[json_start:json_end].strip()

    # Se non è presente il blocco, cerca il JSON direttamente nel testo
    if not json_command and "{\"chart\":" in content:
        json_start = content.find("{\"chart\":")
        brace_count = 1
        for i in range(json_start + 1, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if brace_count == 0:
            json_command = content[json_start:json_end].strip()

    if json_command:
        result = generate_graph(json_command)
        result_message = AIMessage(
            content=f"Ho generato il grafico: {result}"
        )
        return {"messages": [result_message]}

    error_message = AIMessage(
        content="Non sono riuscito a estrarre un comando JSON valido per generare il grafico."
    )
    return {"messages": [error_message]}

@tool
def generate_graph(json_input: str) -> str:
    """
    Genera un grafico usando matplotlib in modo generalizzato.
    
    Il parametro json_input deve essere una stringa JSON con il seguente formato:
    
    {
        "chart": "bar",           # Tipo di grafico: 'bar', 'line', 'pie', 'scatter'
        "data": [
            {"label": "A", "value": 10},
            {"label": "B", "value": 20}
        ],
        "title": "Titolo del grafico",      # Opzionale
        "xlabel": "Etichetta X",             # Opzionale
        "ylabel": "Etichetta Y",             # Opzionale
        "figsize": [8, 6]                    # Opzionale: dimensione della figura (larghezza, altezza)
    }
    
    Salva il grafico come 'graph.png' e restituisce un messaggio di conferma.
    """
    try:
        config = json.loads(json_input)
        chart_type = config.get("chart")
        data_points = config.get("data")
        title = config.get("title", "Grafico Generato")
        xlabel = config.get("xlabel", "X")
        ylabel = config.get("ylabel", "Y")
        figsize = config.get("figsize", [8, 6])
    except Exception as e:
        return f"Errore nel parsing del JSON: {str(e)}"
    
    # Validazione dei dati obbligatori
    if not chart_type or not data_points:
        return "Errore: i campi 'chart' e 'data' sono obbligatori."
    
    if not isinstance(data_points, list) or not all(
        isinstance(dp, dict) and "label" in dp and "value" in dp for dp in data_points
    ):
        return "Errore: il campo 'data' deve essere una lista di oggetti con chiavi 'label' e 'value'."
    
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
        # Per lo scatter plot usiamo gli indici come coordinate x
        plt.scatter(range(len(values)), values)
        plt.xticks(range(len(labels)), labels)
    else:
        return "Errore: tipo di grafico non supportato. Usa 'bar', 'line', 'pie' o 'scatter'."
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig("graph.png")
    plt.close()
    
    return "Grafico generato e salvato come 'graph.png'."

# -------- AGENT NETWORK DEFINITION --------

members = ["graph creator", "data analyst"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

orchestrator_system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class TaskAssignment(BaseModel):
    """Schema for assigning a task to an agent."""
    next: Literal[*options] = Field(..., description="The task to be assigned to an agent.")


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]

class State(MessagesState):
    next: str

def orchestrator_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": orchestrator_system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(TaskAssignment).invoke(messages)
    goto = response.next
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


graph_creator_agent = create_react_agent(
    llm, tools=[
        execute_graph_from_response, generate_graph
    ], prompt="You are a helpful AI assistant who must provide a JSON command to generate a graph using matplotlib. "
        "Return a JSON in the following format: {\"chart\": \"bar\", \"data\": [{\"label\": \"A\", \"value\": 10}, ...]} "
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


data_analyst_agent = create_react_agent(llm, tools=[query_bigquery], 
prompt = "You are a helpful AI assistant who must provide a SQL BigQuery query "
        "to analyze data as the user requests. You can query data from: "
        "`qwiklabs-gcp-03-d68fba73ee2d.sales`. "
        "In BigQuery we have the following table (sales_online): "
        "Order_ID (INTEGER), Date (DATE), Customer (STRING), Product (STRING), "
        "Quantity (INTEGER), Price_per_unit (INTEGER), Total_Amount (INTEGER), "
        "Payment_Method (STRING).")


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


workflow = StateGraph(State)
workflow.add_edge(START, "orchestrator")
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("graph creator", graph_creator_node)
workflow.add_node("data analyst", data_analyst_node)
agent = workflow.compile()


# Ottieni l'immagine in formato PNG
png_data = agent.get_graph().draw_mermaid_png()

# Salva l'immagine in un file chiamato "graph.png"
with open("graph_2.png", "wb") as file:
    file.write(png_data)

print("Il grafico è stato salvato come 'graph.png'")