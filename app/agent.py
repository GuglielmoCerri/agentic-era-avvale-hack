import json
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from google.cloud import bigquery
from langchain_core.messages import AIMessage

LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# Initialize a BigQuery client
client = bigquery.Client()

# 1. Define tools
@tool
def search(query: str) -> str:
    """Simula una ricerca web per ottenere informazioni sul meteo."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

@tool
def query_bigquery(query_str: str) -> list[dict]:
    """Esegue una query SQL su BigQuery.
   
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
   
    # Esegui la query
    try:
        query_job = client.query(sql_query)
        results = query_job.result()
        # Converte i risultati in una lista di dizionari
        rows = [dict(row) for row in results]
        return rows
    except Exception as e:
        return [{"error": f"Errore nell'esecuzione della query: {str(e)}"}]

tools = [query_bigquery]

# 2. Set up the language model
llm = ChatVertexAI(
    model=LLM,
    location=LOCATION,
    temperature=0,
    max_tokens=1024,
    streaming=True
).bind_tools(tools)

# 3. Define workflow components
def should_continue(state: MessagesState) -> str:
    """Determina se continuare con i tool o terminare la conversazione."""
    last_message = state["messages"][-1]
    
    # Se l'ultimo messaggio contiene chiamate a tool, vai al nodo tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Controlla se c'è un comando di esecuzione nella risposta del modello
    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content
        if isinstance(content, str) and ("```json" in content or "{\"query\":" in content):
            return "execute_query"
    
    # Altrimenti termina
    return END

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Chiama il modello di linguaggio e restituisce la risposta."""
    system_message = (
        "You are a helpful AI assistant who must provide a SQL BigQuery query "
        "to analyze data as the user requests. You can query data from: "
        "`qwiklabs-gcp-03-d68fba73ee2d.sales`. "
        "Return a JSON in the following format: {\"query\": \"SELECT * FROM ...\"} "
        "Do not use tool calls directly, just return the JSON."
        "in bigquery we have the following tables:"
        "sales_online:"
        "Order_ID: un identificativo numerico dell'ordine (tipo INTEGER)."
        "Date: la data in cui è stato effettuato l'ordine (tipo DATE)."
        "Customer: il nome del cliente (tipo STRING)."
        "Product: il prodotto ordinato (tipo STRING)."
        "Quantity: la quantità del prodotto ordinato (tipo INTEGER)."
        "Price_per_unit: il prezzo per unità del prodotto (tipo INTEGER)."
        "Total_Amount: l'importo totale dell'ordine (tipo INTEGER)."
        "Payment_Method: il metodo di pagamento utilizzato (tipo STRING)."
        
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    # Propaga il RunnableConfig per supportare lo streaming della risposta.
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

def execute_query_from_response(state: MessagesState) -> dict:
    """Estrae ed esegue la query dall'ultimo messaggio"""
    last_message = state["messages"][-1]
    content = last_message.content
    
    # Estrai la query JSON dalla risposta
    json_query = None
    
    # Cerca di estrarre JSON da codice markdown
    if "```json" in content:
        # Estrai il contenuto tra ```json e ```
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        if json_end > json_start:
            json_query = content[json_start:json_end].strip()
    
    # Se non è in formato markdown, cerca direttamente
    if not json_query and "{\"query\":" in content:
        # Cerca di estrarre il JSON che inizia con {"query":
        json_start = content.find("{\"query\":")
        # Trova la fine del JSON (la parentesi graffa di chiusura)
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
        # Esegui la query utilizzando lo strumento query_bigquery
        result = query_bigquery(json_query)
        
        
        result_message = AIMessage(
            content=f"Ho eseguito la query e ecco i risultati:\n\n{result}"
        )
        return {"messages": [result_message]}
    
    error_message = AIMessage(
        content="Non sono riuscito a estrarre una query SQL valida dalla risposta."
    )
    return {"messages": [error_message]}

# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("execute_query", execute_query_from_response)
workflow.set_entry_point("agent")

# 5. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
workflow.add_edge("execute_query", END)

# 6. Compile the workflow
agent = workflow.compile()

# Esempio di utilizzo:
# response = agent.invoke({"messages": [HumanMessage(content="Quante vendite abbiamo avuto per ogni prodotto nel 2023?")]})
# print(response)