"""Deep Dive Agent for payment investigation and root cause analysis"""

from google.adk.agents import BaseAgent
from google.adk.tools import BigQueryTool
from . import prompt

MODEL = "gemini-2.5-pro"

deep_dive_agent = BaseAgent(
    model=MODEL,
    name="deep_dive_agent",
    instruction=prompt.DEEP_DIVE_AGENT_PROMPT,
    output_key="deep_dive_analysis_output",
    tools=[BigQueryTool(project_id="crowncoins-casino")],
)