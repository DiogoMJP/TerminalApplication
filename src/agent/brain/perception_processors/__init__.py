from __future__	import annotations

from src.agent.brain.perception_processors.perception_processor							import PerceptionProcessor

from typing	import Any, NotRequired, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent	import Agent
	from src.food	import Food


type Sound = tuple[int, tuple[int, int], tuple[int, ...]]

class EnvironmentData(TypedDict):
	food_list	: NotRequired[list[Food]]
	agent_list	: NotRequired[list[Agent]]
	sound_list	: NotRequired[list[Sound]]


def create_perception_processor(perception_processor_type: str, params: dict[str, Any]) -> PerceptionProcessor:
	try:
		if perception_processor_type == "default-perception-processor":
			return PerceptionProcessor.create_from_parameters(params)
		else:
			raise Exception(f"Invalid perception processor type: {perception_processor_type}")
	except Exception as e: raise

def get_perception_processor_parameters(perception_processor_type: str) -> tuple[tuple[str, type], ...]:
	if perception_processor_type == "default-perception-processor":
		return PerceptionProcessor.get_parameters()
	else:
		raise Exception(f"Invalid perception processor type: {perception_processor_type}")

def load_perception_processor_from_data(data: dict[str, Any]) -> PerceptionProcessor:
	if data["type"] == "default-perception-processor":
		return PerceptionProcessor.load_from_data(data)
	else:
		raise Exception(f"Invalid perception processor type: {data['type']}")

def get_perception_processor_types() -> list[str]:
	return ["default-perception-processor"]