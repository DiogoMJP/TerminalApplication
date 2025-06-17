from __future__	import annotations

from src.utils	import CreatableFromParameters

import numpy as np
from abc	import abstractmethod
from math	import sqrt
from typing	import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent	import Agent
	from src.food	import Food


class PerceptionProcessor(CreatableFromParameters):
	def __init__(self, n_input: int):
		self.n_input: int = n_input
	
	def get_closest_food(self, state: Dict[str, Any], food_list: List[Food]) -> tuple[Food | None, float]:
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
		self, state: Dict[str, Any], perception_distance: int, food_list: List[Food], agent_list: List[Agent]
	) -> Tuple[float, ...]:
		raise NotImplementedError(f"{self.__class__.__name__}: process_input method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> Dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")
	
	@staticmethod
	@abstractmethod
	def load_from_data(data: dict[str, Any]) -> 'PerceptionProcessor':
		raise NotImplementedError(f"{__class__.__name__}: load_from_data method must be implemented in subclasses")