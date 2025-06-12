from src.agent.brain.perception_processors.perception_processor						import PerceptionProcessor
from src.agent.brain.perception_processors.food_distance_perception_processor		import FoodDistancePerceptionProcessor
from src.agent.brain.perception_processors.food_agent_distance_perception_processor	import FoodAgentDistancePerceptionProcessor
from src.agent.brain.perception_processors.normalised_input_perception_processor	import NormalisedInputPerceptionProcessor

from typing	import Any


def create_perception_processor(perception_processor_type: str, params: dict[str, Any]) -> PerceptionProcessor:
	try:
		if perception_processor_type == "food-distance-perception-processor":
			return FoodDistancePerceptionProcessor.create_from_parameters(params)
		elif perception_processor_type == "food-agent-distance-perception-processor":
			return FoodAgentDistancePerceptionProcessor.create_from_parameters(params)
		elif perception_processor_type == "normalised-input-perception-processor":
			return NormalisedInputPerceptionProcessor.create_from_parameters(params)
		else:
			raise Exception(f"Invalid perception processor type: {perception_processor_type}")
	except Exception as e: raise

def get_perception_processor_parameters(perception_processor_type: str) -> tuple[str, ...]:
	if perception_processor_type == "food-distance-perception-processor":
		return FoodDistancePerceptionProcessor.get_parameters()
	elif perception_processor_type == "food-agent-distance-perception-processor":
		return FoodAgentDistancePerceptionProcessor.get_parameters()
	elif perception_processor_type == "normalised-input-perception-processor":
		return NormalisedInputPerceptionProcessor.get_parameters()
	else:
		raise Exception(f"Invalid perception processor type: {perception_processor_type}")

def load_perception_processor_from_data(data: dict[str, Any]) -> PerceptionProcessor:
	if data["type"] == "food-distance-perception-processor":
		return FoodDistancePerceptionProcessor()
	elif data["type"] == "food-agent-distance-perception-processor":
		return FoodAgentDistancePerceptionProcessor()
	elif data["type"] == "normalised-input-perception-processor":
		return NormalisedInputPerceptionProcessor()
	else:
		raise Exception(f"Invalid perception processor type: {data['type']}")