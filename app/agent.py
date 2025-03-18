import json
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from google.cloud import bigquery
 
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
    except (json.JSONDecodeError, KeyError) as e:
        return [{"error": f"Formato di input non valido: {e}"}]
   
    # Esegui la query
    query_job = client.query(sql_query)
    results = query_job.result()
    # Converte i risultati in una lista di dizionari
    rows = [dict(row) for row in results]
    return rows
 
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
    return "tools" if last_message.tool_calls else END
 
def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Chiama il modello di linguaggio e restituisce la risposta."""
    system_message = (
        "You are a helpful AI assistant who must provide a SQL BigQuery query "
        "to analyze data as the user requests. You can query data from: "
        "`qwiklabs-gcp-03-3716b0e74aea.data.sales_online`. "
        "Return a JSON in the following format: {\"query\": \"SELECT * FROM ...\"}"
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    # Propaga il RunnableConfig per supportare lo streaming della risposta.
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}
 
# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
 
# 5. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
 
# 6. Compile the workflow
agent = workflow.compile()
 
