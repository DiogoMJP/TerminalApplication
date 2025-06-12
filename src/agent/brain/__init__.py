from src.agent.brain.brain					import Brain
from src.agent.brain.perception_processors	import create_perception_processor
from src.agent.brain.neat_brain				import NeatBrain

from typing	import Any


def create_brain(brain_type: str, params: dict[str, Any]) -> Brain:
	try:
		if "perception-processor-type" not in params.keys():
			raise Exception("Missing required parameter: perception-processor-type")
		params["perception-processor"] = create_perception_processor(
			params["perception-processor-type"], params
		)
		if brain_type == "neat-brain":
			return NeatBrain.create_from_parameters(params)
		else:
			raise Exception(f"Invalid brain type: {brain_type}")
	except Exception as e: raise
	
def get_brain_parameters(brain_type: str) -> tuple[str, ...]:
	if brain_type == "neat-brain":
		return NeatBrain.get_parameters()
	else:
		raise Exception(f"Invalid brain type: {brain_type}")

def load_brain_from_data(data: dict[str, Any]) -> Brain:
	if data["type"] == "neat-brain":
		return NeatBrain.load_from_data(data)
	else:
		raise Exception(f"Invalid brain type: {data['type']}")