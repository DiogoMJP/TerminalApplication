from src.agent.brain.brain					import Brain
from src.agent.brain.neat_brain				import NeatBrain

from typing	import Any


def create_brain(brain_type: str, params: dict[str, Any]) -> Brain:
	try:
		if brain_type == "neat-brain":
			return NeatBrain.create_from_parameters(params)
		else:
			raise Exception(f"Invalid brain type: {brain_type}")
	except Exception as e: raise
	
def get_brain_parameters(brain_type: str) -> tuple[tuple[str, type], ...]:
	if brain_type == "neat-brain":
		return NeatBrain.get_parameters()
	else:
		raise Exception(f"Invalid brain type: {brain_type}")

def load_brain_from_data(data: dict[str, Any]) -> Brain:
	if data["type"] == "neat-brain":
		return NeatBrain.load_from_data(data)
	else:
		raise Exception(f"Invalid brain type: {data['type']}")

def get_brain_types() -> list[str]:
	return ["neat-brain"]