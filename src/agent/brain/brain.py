from __future__	import annotations

from src.utils	import CreatableFromParameters, Loadable

from abc	import abstractmethod
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent								import Agent
	from src.agent.brain.perception_processors	import PerceptionProcessor
	from src.food								import Food


class Brain(CreatableFromParameters, Loadable):
	def __init__(self, n_output: int, perception_processor: PerceptionProcessor):
		self.n_output				: int					= n_output
		self.perception_processor	: PerceptionProcessor	= perception_processor

	def get_closest_food(self, state: dict[str, Any], food_list: list[Food]) -> tuple[Food | None, float]:
		return self.perception_processor.get_closest_food(state, food_list)

	def get_perception(
		self, state: dict[str, Any], perception_distance: int, food_list: list[Food], agent_list: list[Agent]
	) -> tuple[float, ...]:
		return self.perception_processor.process_input(state, perception_distance, food_list, agent_list)

	@abstractmethod
	def get_n_nodes(self) -> int:
		raise NotImplementedError(f"{self.__class__.__name__}: get_n_nodes method must be implemented in subclasses")

	@abstractmethod
	def get_action(
		self, state: dict[str, Any], perception_distance: int, food_list: list[Food], agent_list: list[Agent]
	) -> tuple[int, ...]:
		raise NotImplementedError(f"{self.__class__.__name__}: get_action method must be implemented in subclasses")

	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")