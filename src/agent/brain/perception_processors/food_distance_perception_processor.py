from __future__	import annotations

from src.agent.brain.perception_processors	import PerceptionProcessor

from math	import atan2, degrees, hypot
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent	import Agent
	from src.food	import Food


class FoodDistancePerceptionProcessor(PerceptionProcessor):
	def __init__(self):
		super().__init__(2)
	
	def process_input(
		self, state: dict[str, Any], perception_distance: int, food_list: list[Food],
		agent_list: list[Agent], width: int, height: int
	) -> tuple[float, float]:
		x = state["x"]; y = state["y"]; angle = state["angle"]
		output = [180.0, 2.0 * perception_distance, 180.0, 2.0 * perception_distance]
		
		for food in food_list:
			if food.alive:
				dx = food.x - x; dy = food.y - y
				dist = hypot(dx, dy)
				if dist <= perception_distance and dist < output[1]:
					output[0] = (degrees(atan2(dy, dx)) - angle + 180) % 360 - 180
					output[1] = dist
		
		return (output[0], output[1])
	
	def to_dict(self) -> dict[str, Any]:
		return {"type"	: "food-distance-perception-processor"}
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return tuple()
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'FoodDistancePerceptionProcessor':
		return FoodDistancePerceptionProcessor()

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'FoodDistancePerceptionProcessor':
		return FoodDistancePerceptionProcessor()