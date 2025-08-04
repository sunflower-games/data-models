
from google.adk.agents import BaseAgent
from . import prompt

MODEL = "gemini-2.5-pro"

visualization_manager_agent = BaseAgent(
    model=MODEL,
    name="visualization_manager_agent",
    instruction=prompt.VISUALIZATION_MANAGER_PROMPT,
    output_key="visualization_output",
    tools=[],  # No external tools - works with session state data
)