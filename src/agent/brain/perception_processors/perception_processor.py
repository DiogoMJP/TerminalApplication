from __future__	import annotations

from src.utils	import CreatableFromParameters

from abc	import abstractmethod
from typing	import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent	import Agent
	from src.food	import Food


class PerceptionProcessor(CreatableFromParameters):
	def __init__(self, n_input: int):
		self.n_input: int = n_input

	@abstractmethod
	def process_input(
		self, state: Dict[str, Any], perception_distance: int, food_list: List[Food], agent_list: List[Agent]
	) -> Tuple[Any, ...]:
		raise NotImplementedError(f"{self.__class__.__name__}: process_input method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> Dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")