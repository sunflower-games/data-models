"""Main KPI Agent for payment analytics using BigQuery"""

from google.adk.agents import BaseAgent
from google.adk.tools import BigQueryTool
from . import prompt

MODEL = "gemini-2.5-pro"

main_kpi_agent = BaseAgent(
    model=MODEL,
    name="main_kpi_agent",
    instruction=prompt.MAIN_KPI_AGENT_PROMPT,
    output_key="main_kpi_analysis_output",
    tools=[BigQueryTool(project_id="crowncoins-casino")],
)