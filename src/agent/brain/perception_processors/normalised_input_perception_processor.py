from __future__	import annotations

from src.agent.brain.perception_processors	import PerceptionProcessor

from math	import atan2, degrees, hypot
from typing	import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import EnvironmentData


class NormalisedInputPerceptionProcessor(PerceptionProcessor):
	def __init__(self):
		super().__init__(6)

	def process_input(
    	self, state: dict[str, Any], perception_distance: int, width: int, height: int,
		**environment_data: Unpack[EnvironmentData]
	) -> tuple[float, float, float, float, float, float]:
		if "food_list" not in environment_data:
			raise Exception(f"{self.__class__.__name__}: Missing required environment data: food_list")
		if "agent_list" not in environment_data:
			raise Exception(f"{self.__class__.__name__}: Missing required environment data: agent_list")
		food_list = environment_data["food_list"]
		agent_list = environment_data["agent_list"]

		x = state["x"]; y = state["y"]; angle = state["angle"]
		output = [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
		
		for food in food_list:
			if food.alive:
				dx = food.x - x; dy = food.y - y
				dist = hypot(dx, dy) / perception_distance
				if dist <= 1 and dist < output[2]:
					output[0] = 1.0
					output[1] = ((degrees(atan2(dy, dx)) - angle + 180) % 360 - 180) / 180
					output[2] = dist
		
		for agent in agent_list:
			if agent.alive:
				agent_x = agent.get_from_state("x"); agent_y = agent.get_from_state("y")
				dx = agent_x - x; dy = agent_y - y
				dist = hypot(dx, dy) / perception_distance
				if dist <= 1 and dist < output[4]:
					output[3] = 1.0
					output[4] = ((degrees(atan2(dy, dx)) - angle + 180) % 360 - 180) / 180
					output[5] = dist
		
		return (output[0], output[1], output[2], output[3], output[4], output[5])

	def to_dict(self) -> dict[str, Any]:
		return {"type"	: "normalised-input-perception-processor"}
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return tuple()
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'NormalisedInputPerceptionProcessor':
		return NormalisedInputPerceptionProcessor()
	
	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'NormalisedInputPerceptionProcessor':
		return NormalisedInputPerceptionProcessor()