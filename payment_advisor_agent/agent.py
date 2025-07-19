"""Payment Advisor: Comprehensive payment data analysis system"""

from google.adk.agents import LlmAgent
from . import prompt
from .sub_agents.main_kpi_agent import main_kpi_agent
from .sub_agents.deep_dive_agent import deep_dive_agent
from .sub_agents.visualization_manager import visualization_manager_agent

MODEL = "gemini-2.5-pro"

payment_advisor = LlmAgent(
    name="payment_advisor",
    model=MODEL,
    description=(
        "Coordinate comprehensive payment analytics by orchestrating "
        "specialized analysis agents. Analyze KPIs, investigate anomalies and patterns, "
        "find correlations across payment data, and create actionable insights and recommendations "
        "for the financial department."
    ),
    instruction=prompt.PAYMENT_ADVISOR_PROMPT,
    output_key="payment_advisor_output",      
    sub_agents=[
        main_kpi_agent,
        deep_dive_agent,
        visualization_manager_agent,
    ],
)

root_agent = payment_advisor 