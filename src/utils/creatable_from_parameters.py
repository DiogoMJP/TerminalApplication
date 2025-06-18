from abc import ABC, abstractmethod
from typing import Any


class CreatableFromParameters(ABC):
	@staticmethod
	@abstractmethod
	def get_parameters() -> tuple[str, ...]:
		raise NotImplementedError(f"{__class__.__name__}: 'get_parameters' must be implemented in subclasses")
	
	@staticmethod
	@abstractmethod
	def create_from_parameters(params: dict[str, Any]) -> 'CreatableFromParameters':
		raise NotImplementedError(f"{__class__.__name__}: 'create_from_parameters' must be implemented in subclasses")