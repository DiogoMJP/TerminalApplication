from abc	import ABC, abstractmethod
from typing	import Any


class Loadable(ABC):
	@staticmethod
	@abstractmethod
	def load_from_data(data: dict[str, Any]) -> 'Loadable':
		raise NotImplementedError(f"{__class__.__name__}: load_from_data method must be implemented in subclasses")