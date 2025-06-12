from src.utils	import CreatableFromParameters

from abc	import abstractmethod
from typing import Any


class NeuralNetwork(CreatableFromParameters):
	def __init__(self):
		pass

	@abstractmethod
	def activate(self, inputs: tuple[float]) -> list[float]:
		raise NotImplementedError(f"{self.__class__.__name__}: activate method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")
	
	@staticmethod
	@abstractmethod
	def load_from_data(data: dict[str, Any]) -> 'NeuralNetwork':
		raise NotImplementedError(f"{__class__.__name__}: load_from_data method must be implemented in subclasses")