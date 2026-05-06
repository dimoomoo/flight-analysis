from langgraph.graph import StateGraph, END
from typing import TypedDict

# ── State ──────────────────────────────────────────────
class FlightState(TypedDict):
    task: str
    result: str

# ── Agent Nodes ────────────────────────────────────────
def delay_analysis_agent(state: FlightState) -> FlightState:
    result = """
    DELAY ANALYSIS FINDINGS:
    - Best hours to fly: 6AM-11AM (lowest avg departure delay)
    - Worst hours: 6PM-9PM (delays peak in evening)
    - Best days: Tuesday and Wednesday
    - Worst days: Friday and Sunday
    - Best months: September and October
    - Worst months: June, July, December
    """
    return {"task": state["task"], "result": result}

def aircraft_age_agent(state: FlightState) -> FlightState:
    result = """
    AIRCRAFT AGE FINDINGS:
    - Correlation between aircraft age and delay: weak (r ≈ 0.05)
    - Slight positive relationship found via linear regression
    - Aircraft age is NOT a strong predictor of delays
    - Operational factors matter more than plane age
    """
    return {"task": state["task"], "result": result}

def logistic_regression_agent(state: FlightState) -> FlightState:
    result = """
    DIVERSION PREDICTION FINDINGS:
    - Departure delay is the strongest predictor of diversion
    - Arrival delay also positively correlates with diversion
    - Distance has a small negative effect on diversion probability
    - Aircraft age has minimal predictive power
    - Model is consistent across 2005, 2006, 2007
    """
    return {"task": state["task"], "result": result}

def router(state: FlightState) -> str:
    task = state["task"].lower()
    if "age" in task or "older" in task or "plane" in task:
        return "aircraft_age"
    elif "divert" in task or "logistic" in task or "predict" in task:
        return "logistic_regression"
    else:
        return "delay_analysis"

# ── Build the Graph ────────────────────────────────────
def build_graph():
    graph = StateGraph(FlightState)

    graph.add_node("delay_analysis", delay_analysis_agent)
    graph.add_node("aircraft_age", aircraft_age_agent)
    graph.add_node("logistic_regression", logistic_regression_agent)

    graph.set_conditional_entry_point(router, {
        "delay_analysis": "delay_analysis",
        "aircraft_age": "aircraft_age",
        "logistic_regression": "logistic_regression"
    })

    graph.add_edge("delay_analysis", END)
    graph.add_edge("aircraft_age", END)
    graph.add_edge("logistic_regression", END)

    return graph.compile()

# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    tasks = [
        "What are the best times to fly to avoid delays?",
        "Do older planes suffer more delays?",
        "What predicts whether a flight gets diverted?"
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print('='*60)
        result = app.invoke({"task": task, "result": ""})
        print(result["result"])