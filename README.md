# Flight Delay Analysis: LangGraph Agentic Workflow

An agentic AI system built with **LangGraph** that analyses US domestic flight data (2005–2007) using a multi-agent architecture.

## How It Works

A central **router** receives a question and intelligently directs it to the correct specialist agent:

Agent Tasks:

1. Delay Analysis Agent: Identifies best/worst times to fly based on departure delay patterns 
2. Aircraft Age Agent: Analyses correlation between aircraft age and delays 
3. Logistic Regression Agent: Predicts flight diversion probability using regression modelling 

## Key Findings

- **Best time to fly:** Tuesday/Wednesday mornings (6AM–11AM)
- **Worst time to fly:** Friday/Sunday evenings (6PM–9PM)
- **Aircraft age:** Weak predictor of delays (r ≈ 0.05) 
- **Diversion predictors:** Departure delay is the strongest signal and distance has a small negative effect

## Tech Stack

- Python
- LangGraph (StateGraph, conditional routing)
- Pandas, NumPy
