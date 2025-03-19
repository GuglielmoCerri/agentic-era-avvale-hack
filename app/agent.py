import json
import matplotlib.pyplot as plt
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from google.cloud import bigquery

import json
import matplotlib.pyplot as plt
#import pandas as pd
import base64

# Configurazioni iniziali
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# Inizializza il client BigQuery
client = bigquery.Client()

# =======================
# 1. Query Agent: esegue query su BigQuery
# =======================
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

tools_query = [query_bigquery]

# Imposta il modello di linguaggio (lo stesso LLM verrà usato per gli agenti)
llm = ChatVertexAI(
    model=LLM,
    location=LOCATION,
    temperature=0,
    max_tokens=1024,
    streaming=True
).bind_tools(tools_query)


# Nodo per chiamare il modello e generare la query SQL
def call_orchestrator(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = (
        "You are a helpful AI assistant who must orchestrate an Agentic AI solution."
        "You must decide whether to:\n"
        "1. Call the validator, to check if it's possible to then query a certain table from bigquery to retrieve data\n"
        "2. If the user asks for a graph call the agent specialized in generating a graph using data already retrieved\n"
        "3. If else answer to the user with available data and/or graph\n"
        # "You can decompose the request and process them iteratively.\n"
        "Return a JSON in the following format:\n If 1: {\"validator\": \"User request with additional details if needed\"}\n"
        "If 2: {\"graph\": \"User request with details on the graph style\"}\n"
        "If 3: answer to the user directly if none of the above is applicable\n"
        "Do not call any tools directly, just return the JSON."
        )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


def call_validator(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = (
        "You are a helpful AI assistant who must check whether available data can answer the user's question.\n"
        "Analyze the user request and verify if relevant data exists in the following source: `qwiklabs-gcp-03-d68fba73ee2d.sales`.\n"
        "In BigQuery we have the following table sales_online: "
        "Order_ID (INTEGER), Date (DATE), Customer (STRING), Product (STRING), "
        "Quantity (INTEGER), Price_per_unit (INTEGER), Total_Amount (INTEGER), "
        "Payment_Method (STRING)."
        "Return a JSON response in one of the following formats:\n"
        "If data is relevant to answer the user: {\"valid\": \"User request to performm the query\"}\n"
        "If data is not relevant to ansert the user: {\"invalid\": \"User request with additional details on why the data is not sufficient to answer\"}\n"
        "Do not call any tools directly, just return the JSON."
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


def call_finalizer(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = (
        "You are a helpful AI assistant who gives the final response to the user.\n"
        "Answer the user based on the output that you are provided with"
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


# Nodo per chiamare il modello e generare la query SQL
def call_decomposer_queries(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = (
        "You are a helpful AI assistant who must provide a SQL BigQuery query "
        "to analyze data as the user requests. You can query data from: "
        "`qwiklabs-gcp-03-d68fba73ee2d.sales`. "
        "Return a JSON in the following format: {\"query\": \"SELECT * FROM ...\"} "
        "Do not use tool calls directly, just return the JSON. "
        "In BigQuery we have the following table (sales_online): "
        "Order_ID (INTEGER), Date (DATE), Customer (STRING), Product (STRING), "
        "Quantity (INTEGER), Price_per_unit (INTEGER), Total_Amount (INTEGER), "
        "Payment_Method (STRING)."
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

def call_decomposer_graph(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = (
        "You are a helpful AI assistant who must provide a JSON command to generate a graph using matplotlib. "
        "Return a JSON in the following format: {\"chart\": \"bar\", \"data\": [{\"label\": \"A\", \"value\": 10}, ...]} "
        "Optionally, you can include parameters such as 'title', 'xlabel', 'ylabel', and 'figsize' to customize the graph appearance. "
        "Do not call any tools directly, just return the JSON."
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


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
        print(json_query)
        result = query_bigquery(json_query)
        result_message = AIMessage(
            content=f"Ho eseguito la query e ecco i risultati:\n\n{result}"
        )
        return {"messages": [result_message]}
    
    error_message = AIMessage(
        content="Non sono riuscito a estrarre una query SQL valida dalla risposta."
    )
    return {"messages": [error_message]}

def execute_graph_from_response(state: MessagesState) -> dict:
    """
    Estrae il comando JSON per generare il grafico dalla risposta del modello
    e lo esegue tramite il tool generate_graph.
    Cerca il JSON racchiuso tra ```json ... ``` oppure direttamente come stringa.
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

        # TODO sistema
        with open("graph.png", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        image_html = f'<img src="data:image/png;base64,{image_data}>'

        result_message = AIMessage(
            content=f"Ho generato il grafico: {image_html}"
        )
        return {"messages": [result_message]}

    error_message = AIMessage(
        content="Non sono riuscito a estrarre un comando JSON valido per generare il grafico."
    )
    return {"messages": [error_message]}

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

def check_validity(state: MessagesState) -> str:
    """
    Se i dati sono semanticamente rilevanti chiama l'agent incaricato di fare la query.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content
        print(content)
        if isinstance(content, str) and ("```json" in content or "{\"valid\":" in content):
            return "decomposer_queries"
        elif isinstance(content, str) and ("```json" in content or "{\"invalid\":" in content):
            return "finalizer"
    return END


def how_to_continue(state: MessagesState) -> str:
    """
    Se l'ultimo messaggio contiene un comando JSON, si procede con validator o graph,
    altrimenti termina.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content
        if isinstance(content, str) and ("```json" in content or "{\"validator\":" in content and not "{\"graph\":" in content):            
            return "validator"
        elif isinstance(content, str) and ("```json" in content or "{\"graph\":" in content):            
            return "decomposer_graph"
    return END


workflow = StateGraph(MessagesState)
workflow.add_node("orchestrator", call_orchestrator) # agent
workflow.add_node("generate_graph", execute_graph_from_response) # tool
workflow.add_node("decomposer_queries", call_decomposer_queries) # agent
workflow.add_node("finalizer", call_finalizer) # agent
workflow.add_node("validator", call_validator) # agent
workflow.add_node("decomposer_graph", call_decomposer_graph) # agent
workflow.add_node("execute_queries", execute_query) # tool
# ---------------------------------------
workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges("orchestrator", how_to_continue)
workflow.add_conditional_edges("validator", check_validity)
workflow.add_edge("decomposer_queries", "execute_queries")
workflow.add_edge("decomposer_graph", "generate_graph")
workflow.add_edge("execute_queries", "orchestrator")
workflow.add_edge("generate_graph", "finalizer")
workflow.add_edge("finalizer", END)

agent = workflow.compile()
# Ottieni l'immagine in formato PNG
png_data = agent.get_graph().draw_mermaid_png()

# Salva l'immagine in un file chiamato "graph.png"
with open("graph_.png", "wb") as file:
    file.write(png_data)

print("Il grafico è stato salvato come 'graph.png'")