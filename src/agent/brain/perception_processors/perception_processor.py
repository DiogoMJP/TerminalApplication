from __future__	import annotations

from src.utils	import CreatableFromParameters, Loadable

from abc	import abstractmethod
from math	import sqrt
from typing	import Any, Tuple, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import EnvironmentData
	from src.food								import Food


class PerceptionProcessor(CreatableFromParameters, Loadable):
	def __init__(self, n_input: int):
		self.n_input: int = n_input
	
	def get_closest_food(self, state: dict[str, Any], food_list: list[Food]) -> tuple[Food | None, float]:
		x, y = state["x"], state["y"]

		min_dist_sq = float('inf')
		closest_food = None

		for food in food_list:
			if not food.alive:
				continue
			dx = x - food.x
			dy = y - food.y
			dist_sq = dx * dx + dy * dy
			if dist_sq < min_dist_sq:
				min_dist_sq = dist_sq
				closest_food = food

		if closest_food is None:
			return None, 0.0

		return closest_food, sqrt(min_dist_sq)

	@abstractmethod
	def process_input(
		self, state: dict[str, Any], perception_distance: int, width: int, height: int,
		**environment_data: Unpack[EnvironmentData]
	) -> Tuple[float, ...]:
		raise NotImplementedError(f"{self.__class__.__name__}: process_input method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")