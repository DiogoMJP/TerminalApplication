from __future__	import annotations

from src.utils	import CreatableFromParameters, Loadable

from abc	import abstractmethod
from typing	import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import EnvironmentData


class PerceptionNode(CreatableFromParameters, Loadable):
	def __init__(self, n_output: int):
		self.n_output	: int	= n_output
	
	@abstractmethod
	def process(
    	self, state: dict[str, Any], perception_distance: int, poisonous_detection_distance: int, 
		width: int, height: int, **environment_data: Unpack[EnvironmentData]
	) -> list[float]:
		raise NotImplementedError(f"{self.__class__.__name__}: process method must be implemented in subclasses")

	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")