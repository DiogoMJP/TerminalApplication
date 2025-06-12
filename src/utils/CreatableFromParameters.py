from abc import ABC, abstractmethod
from typing import Any


class CreatableFromParameters(ABC):
	@staticmethod
	@abstractmethod
	def get_parameters() -> tuple[str, ...]:
		raise NotImplementedError("'get_parameters' must be implemented in subclasses")
	
	@staticmethod
	@abstractmethod
	def create_from_parameters(params: dict[str, Any]) -> 'CreatableFromParameters':
		raise NotImplementedError("'create_from_parameters' must be implemented in subclasses")