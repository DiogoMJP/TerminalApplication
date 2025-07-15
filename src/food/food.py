from __future__	import annotations

from src.utils import CreatableFromParameters

from abc	import abstractmethod
from random	import random
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent import Agent
	

class Food(CreatableFromParameters):
	def __init__(
		self, x: int, y: int, eating_number: int, first_time_step: int,
		lifespan: int, detection_radius: int, poisonous_food_rate: float
):
		self.x						: int	= x
		self.y						: int	= y
		self.eating_number			: int	= eating_number
		self.first_time_step		: int	= first_time_step
		self.lifespan				: int	= lifespan
		self.detection_radius		: int	= detection_radius
		self.poisonous_food_rate	: float	= poisonous_food_rate
		self.poisonous				: bool	= random() < poisonous_food_rate

		self.eaten_by		: list[Agent]	= []
		self.alive			: bool			= True
		self.last_time_step	: int			= 0
		self.eaten			: bool			= False

	@abstractmethod
	def simulate(self, time_step: int) -> None:
		raise NotImplementedError(f"{self.__class__.__name__}: simulate method must be implemented in subclasses")

	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")