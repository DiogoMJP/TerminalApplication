from src.agent.agent			import Agent
from src.agent.default_agent	import DefaultAgent

from typing import Any


def create_agent(agent_type: str, params: dict[str, Any]) -> Agent:
	try:
		if agent_type == "default-agent":
			return DefaultAgent.create_from_parameters(params)
		else:
			raise Exception(f"Invalid agent type: {agent_type}")
	except Exception as e: raise

def get_agent_parameters(agent_type: str) -> tuple[str, ...]:
	if agent_type == "default-agent":
		return DefaultAgent.get_parameters()
	else:
		raise Exception(f"Invalid agent type: {agent_type}")