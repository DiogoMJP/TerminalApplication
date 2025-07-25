from src.utils	import CreatableFromParameters, Loadable

from abc	import abstractmethod
from typing import Any


class NeuralNetwork(CreatableFromParameters, Loadable):
	def __init__(self):
		pass

	@abstractmethod
	def activate(self, inputs: tuple[float]) -> list[float]:
		raise NotImplementedError(f"{self.__class__.__name__}: activate method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")