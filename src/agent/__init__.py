from src.agent.agent					import Agent
from src.agent.default_agent			import DefaultAgent
from src.agent.stopped_by_walls_agent	import StoppedByWallsAgent

from typing import Any


def create_agent(agent_type: str, params: dict[str, Any]) -> Agent:
	try:
		if agent_type == "default-agent":
			return DefaultAgent.create_from_parameters(params)
		if agent_type == "stopped-by-walls-agent":
			return StoppedByWallsAgent.create_from_parameters(params)
		else:
			raise Exception(f"Invalid agent type: {agent_type}")
	except Exception as e: raise

def get_agent_parameters(agent_type: str) -> tuple[str, ...]:
	if agent_type == "default-agent":
		return DefaultAgent.get_parameters()
	if agent_type == "stopped-by-walls-agent":
		return StoppedByWallsAgent.get_parameters()
	else:
		raise Exception(f"Invalid agent type: {agent_type}")

def get_agent_types() -> list[str]:
	return [
		"default-agent",
		"stopped-by-walls-agent"
	]