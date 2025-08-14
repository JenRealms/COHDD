"""
data_collector.py

This module defines the task_status response class and initializes the LLM and prompt for data collection tasks.
Handles the orchestration of data collection prompts and status tracking.
"""
import pprint, json, datetime, warnings
from langchain_aws import ChatBedrock
from typing import Optional, List, TypedDict, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ValidationError
from .data_collector_prompt import DataCollectorPrompt, QueryRevisePrompt
from .data_collect_tool import query_athena_to_markdown
from langgraph.graph import StateGraph, MessageGraph, START, END

_sql_query_tracker = {}

def record_sql_query_global(chain_name: str, sql_query: str, status: str = "pending", error: Optional[str] = None):
    """Record SQL query globally for debugging"""
    if chain_name not in _sql_query_tracker:
        _sql_query_tracker[chain_name] = []
    
    query_record = {
        "sql_query": sql_query,
        #"timestamp": datetime.now().isoformat(),
        "status": status,
        "error": error
    }
    
    _sql_query_tracker[chain_name].append(query_record)

def get_sql_queries(chain_name: Optional[str] = None):
    """Get recorded SQL queries"""
    if chain_name:
        return _sql_query_tracker.get(chain_name, [])
    return _sql_query_tracker

# Node names
sql_gen_name = "sql_gen"
data_collect_name = "data_collect"
revise_sql_name = "revise_sql"

# State class
class task_status(BaseModel):
    """
    Represents the status of a data collection task.

    Attributes:
        task (str): The data collection task that user wants to achieve.
        sql_query (str): The SQL query generated to address the data collection task.
        collected_data (str): The collected data in the Markdown format.
    """
    task: str = Field(
        ...,
        description="The data collection task that user wants to achieve"
        )

    sql_query: Optional[str] = None
    collected_data: Optional[str] = None
    max_retries: int = 5
    current_retry: int = 0


# Node functions
def gen_sql_query(state: task_status) -> task_status:
    """
    Main execution for initializing the LLM and prompt, and printing the prompt template.
    Handles errors gracefully and logs them.
    """
    #print("Starting SQL query generation...")
    #print(f"Task received: {state.task}")
    
    try:
        llm = ChatBedrock(
            #model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_kwargs={"temperature": 0.0},
        )
        #print("LLM initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize ChatBedrock LLM: {e}")
        return state  # Return state instead of None

    try:
        prompt = DataCollectorPrompt().get_prompt(state.task, task_status.__name__)
        if prompt is None:
            print("[ERROR] Prompt construction returned None")
            return state
        #print("Prompt constructed successfully")
        #print("*****Prompt:\n", prompt)
    except Exception as e:
        print(f"[ERROR] Failed to initialize or print DataCollectorPrompt: {e}")
        return state  # Return state instead of None
    
    try:
        validator = PydanticToolsParser(tools=[task_status])
        #test = prompt | llm
        #print("Test:", test)
        get_data_chain = prompt | llm.bind_tools(tools=[task_status]) | validator
        #print("Chain constructed successfully")
        
        response = get_data_chain.invoke({"task": state.task})
        #print("Chain invoked successfully")
        #print("Raw response:", response)
        
        if isinstance(response, list) and len(response) > 0:
            result = response[0]
            state.task = result.task
            state.sql_query = result.sql_query
            #print("*****Result:\n", response)
            #print("*****SQL Query:\n", state.sql_query)
            return state
        else:
            print(f"[ERROR] Unexpected response format: {response}")
            return state
    except Exception as e:
        print(f"[ERROR] Failed during chain execution: {e}")
        return state

def revise_sql_query(state: task_status) -> task_status:
    """
    Main execution for initializing the LLM and prompt, and printing the prompt template.
    Handles errors gracefully and logs them.
    """
    #print("-----", revise_sql_name, "-----")
    try:
        llm = ChatBedrock(
            #model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            model_kwargs={"temperature": 0.0},
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize ChatBedrock LLM: {e}")
        return

    try:
        prompt = QueryRevisePrompt().get_prompt(state.task, state.sql_query, state.collected_data, task_status.__name__)
        #print(prompt)
    except Exception as e:
        print(f"[ERROR] Failed to initialize or print DataCollectorPrompt: {e}")
        return
    
    validator = PydanticToolsParser(tools=[task_status])

    get_data_chain = prompt | llm.bind_tools(tools=[task_status]) | validator
    response = get_data_chain.invoke({"task": state.task})
    if isinstance(response, list) and len(response) > 0:
        result = response[0]
        state.current_retry += 1  # Revise status to indicate the number of retries
        state.sql_query = result.sql_query
        #print("Current Retry:", state.current_retry)
        #print("Revised SQL Query:\n", state.sql_query)
        return state
    else:
        raise ValueError("Unexpected response format from get_data_chain")

def collect_data(state: task_status) -> task_status:
    #print("-----", data_collect_name, "-----")
    sql_query = state.sql_query

    # Record SQL query globally
    record_sql_query_global("data_collector", sql_query, status="pending")

    try:
        result = query_athena_to_markdown.invoke(sql_query)
        #result = query_athena_to_markdown(sql_query)
        if result is None:
            state.collected_data = "No data found for the query."
        else:
            state.collected_data = result
        # Update status to success
        record_sql_query_global("data_collector", sql_query, status="success")

    except Exception as e:
        # Capture the error message
        state.collected_data = f"ERROR: {str(e)}"
        error_msg = f"ERROR: {str(e)}"
        state.collected_data = error_msg

        # Update status to error
        record_sql_query_global("data_collector", sql_query, status="error", error=error_msg)

    return state

def route_to_revise(state: task_status) -> task_status:
    if 'error' in state.collected_data.lower() and state.current_retry < state.max_retries:
        return "revise_sql"
    elif state.collected_data is not None and state.current_retry >= state.max_retries and 'error' in state.collected_data.lower():
        return "end"
    else:
        return "end"

def data_collection(task: str) -> str:
    """
    Description:
        This tool automates the process of generating, executing, and revising SQL queries to fulfill a COH data collection request.
        It uses a stateful agent graph to:
            1. Generate an initial SQL query based on the user's data collection task.
            2. Execute the query and collect results.
            3. If an error occurs, automatically revise the SQL query using the error message and retry, up to a maximum number of retries.
        The process continues until data is successfully collected or the retry limit is reached.

    Args:
        task (str): 
            A natural language description of the data collection request. 
            Example: "What is the COH for station DBM3 between 2025-3-9 and 2025-06-05?"

    Returns:
        {
            "status": "success" or "error",
            "collected_data": the data inquired by the query in the Markdown format (str)
            "error_message": the error message if the data collection fails (str)
        }
    """
    
    # GRAPH
    graph = StateGraph(task_status)

    graph.add_node(sql_gen_name, gen_sql_query)
    graph.add_node(data_collect_name, collect_data)
    graph.add_node(revise_sql_name, revise_sql_query)

    graph.add_edge(START, sql_gen_name)
    graph.add_edge(sql_gen_name, data_collect_name)
    graph.add_conditional_edges(
        data_collect_name, 
        route_to_revise, 
        {
           "revise_sql": revise_sql_name,
           "end": END
        }
    )
    graph.add_edge(data_collect_name, END)
    graph.add_edge(revise_sql_name, data_collect_name)

    # Compile the graph
    app = graph.compile()
    #app.get_graph().draw_mermaid_png(output_file_path="data_collector_graph.png")

    # Invoke the graph
    try:
        result = app.invoke(task_status(task=task))
        final_data = result["collected_data"]
        return {
            "status": "success",
            "collected_data": final_data,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }
    #print("Task:\n", result["task"])
    #print("SQL Query:\n", result["sql_query"])
    #print("Collected Data:\n", result["collected_data"])
    
    


def data_collector_subgraph() -> StateGraph:
    graph = StateGraph(task_status)

    graph.add_node(data_collect_name, collect_data)
    graph.add_node(revise_sql_name, revise_sql_query)
    graph.add_node(sql_gen_name, gen_sql_query)

    graph.add_edge(START, data_collect_name)
    graph.add_edge(data_collect_name, revise_sql_name)
    graph.add_edge(revise_sql_name, sql_gen_name)
    graph.add_edge(sql_gen_name, END)

    return graph.compile()




task = """ 
Please extract the following data for station DBM3 on target date 2025-06-05 and the previous 30 days (2025-05-06 to 2025-06-05):

**Core Capacity and Volume Metrics:**
- station_code, ofd_date, capped_out_hours
- w3_capacity_ask (3-week forecast volume)
- w1_cap_target (1-week capacity target)
- w1_caps (1-week actual capacity locked)
- daily_updated_cap_target (DUCT - final daily target)
- d1_caps (actual planned capacity for delivery day)
- latest_slammed_volume (actual packages processed)
- latest_utilization (capacity utilization percentage)

**Capacity Component Analysis:**
- w1_utr, d1_utr (UTR capacity: 1-week vs delivery day)
- w1_otr, d1_otr (OTR capacity: 1-week vs delivery day)
- d1_mech (mechanical capacity)
- d1_constraint (primary constraint type)
- d1_vs_w1_utr_change (UTR capacity change percentage)

**Root Cause Attribution:**
- main_constraint (primary COH root cause)
- main_constraint_bucket (root cause category)
- latest_caps_vs_duct (capacity vs DUCT comparison)
- caps_change (tactical capacity changes W-1 to OFD)
- primary_main_reason (reason for capacity changes)
- tactical_caps_root_cause (aggregated tactical change reason)

**External Factors:**
- weather_signal, weather_tier, ofd_weather_flag, prior_3_ofd_weather_flag
- backlog_flag, upstream_backlog, instation_backlog, total_backlog, vbl_eod
- cf_exclusion_flag, co_exclusion_flag, exclusion_reason
- manual_cap_down (manual capacity reductions)

**Calculated Deltas Needed:**
- Delta between w3_capacity_ask and daily_updated_cap_target
- Delta between w1_cap_target and daily_updated_cap_target  
- Delta between d1_caps and daily_updated_cap_target
- Percentage change: (d1_utr - w1_utr) / w1_utr * 100
- Percentage change: (d1_otr - w1_otr) / w1_otr * 100

"""

task2 = """
Please provide the following data elements for station DBM3 with target date 2025-06-05 and the previous 30 days (2025-05-06 to 2025-06-05):

1. **Capacity Planning Metrics:**
   - `w1_cap_target` (Week-1 capacity target)
   - `d1_caps` (Day-1 actual capacity)
   - `duct` (Daily Updated Capacity Target)
   - `d1_utr` (Day-1 Under-the-Roof capacity)
   - `d1_otr` (Day-1 On-the-Road capacity)

2. **Utilization and Volume Metrics:**
   - `latest_utilization` (Current capacity utilization)
   - `latest_slammed_volume` (Actual processed volume)
   - `capped_out_hours` (COH value)
   - `rolling_21_day_caps` (Rolling capacity for weighted calculations)

3. **Constraint and Weather Indicators:**
   - `main_constraint` (Primary constraint type)
   - `weather_tier` and `weather_signal`
   - `ofd_weather_flag` (Weather impact on delivery date)
   - `prior_3_ofd_weather_flag` (Previous weather impacts)

4. **Backlog and Exclusion Flags:**
   - `upstream_backlog` and `instation_backlog`
   - `backlog_flag`
   - `cf_exclusion_flag` and `co_exclusion_flag`
   - `exclusion_reason`

5. **Tactical Changes:**
   - `intra_week_cap_reduction_flag`
   - `manual_cap_down` values
   - Any capacity adjustment indicators

6. **Historical Context:**
   - Same metrics for the past 30 days to identify patterns
   - Filter: station_code = 'DBM3', cycle = 'CYCLE_1', country_code in ('US', 'CA')

### Analysis Framework Ready for Implementation

Once the data is available, I will analyze the following key areas:

**1. UTR Flex-Up Analysis:**
- Compare DUCT vs. planned UTR capacity
- Verify if UTR increased by minimum 10% when demand exceeded plan
- Identify UTR-related constraints

**2. OTR Flex-Up Analysis:**
- Compare daily volume target vs. 3-week forecast OTR capacity
- Assess OTR flex-up performance against 10% threshold
- Evaluate delivery capacity constraints

**3. Root Cause Categorization:**
- Demand Signal issues (D-1 or W-1)
- Capacity Planning failures
- Resource Constraints (UTR/OTR/Mechanical)
- Weather Impact assessment
- Backlog Management issues

**4. Historical Pattern Analysis:**
- Identify similar COH events in past 30 days
- Assess recurring constraint patterns
- Evaluate station's flex-up capability trends
"""

#result = data_collection("Tell me the COH for station DBM3 on target date 2025-07-29.")
#print("*****Result:\n", result)
