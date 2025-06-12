from __future__	import annotations

from src.agent.brain.perception_processors	import PerceptionProcessor

from math	import atan2, degrees, sqrt
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent	import Agent
	from src.food	import Food


class FoodDistancePerceptionProcessor(PerceptionProcessor):
	def __init__(self):
		super().__init__(2)
	
	def process_input(
		self, state: dict[str, Any], perception_distance: int, food_list: list[Food], agent_list: list[Agent]
	) -> tuple[Food | None, float, float]:
		x = state["x"]; y = state["y"]
		closest_x = None; closest_y = None; dist = None; closest = None
		
		for food in food_list:
			if food.alive:
				if dist == None:
					dist = (x-food.x)**2 + (y-food.y)**2
					closest_x = food.x; closest_y = food.y
					closest = food
				else:
					d = (x-food.x)**2 + (y-food.y)**2
					if d < dist:
						dist = d
						closest_x = food.x; closest_y = food.y
						closest = food
		
		if dist == None:
			return (None, 500, 500)

		dist = sqrt(dist)
		angle = (degrees(atan2(closest_y - y, closest_x - x)) - state["angle"] + 180) % 360 - 180
		return (closest, angle, dist) if dist < perception_distance else (None, 500, 500)
	
	
	def to_dict(self) -> dict[str, Any]:
		return {"type" : "food-distance-perception-processor"}
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return tuple()
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'FoodDistancePerceptionProcessor':
		return FoodDistancePerceptionProcessor()