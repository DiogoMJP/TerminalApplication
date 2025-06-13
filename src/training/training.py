from __future__	import annotations

from src.utils	import CreatableFromParameters

from abc	import abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain


class Training(CreatableFromParameters):
	def __init__(
		self, n_generations: int, width: int, height: int, n_agents: int, agent_type: str, agents_lifespan: int,
		agents_lifespan_extension: int, food_lifespan: int, perception_distance: int, eating_distance: int,
		eating_number: int, max_time_steps: int, simulation_type: str
	):
		self.n_generations				: int			= n_generations
		self.width						: int			= width
		self.height						: int			= height
		self.n_agents					: int			= n_agents
		self.agent_type					: str			= agent_type
		self.agents_lifespan			: int			= agents_lifespan
		self.agents_lifespan_extension	: int			= agents_lifespan_extension
		self.food_lifespan				: int			= food_lifespan
		self.perception_distance		: int			= perception_distance
		self.eating_distance			: int			= eating_distance
		self.eating_number				: int			= eating_number
		self.max_time_steps				: int			= max_time_steps
		self.simulation_type			: str			= simulation_type
		self.brain						: Brain|None	= None
	
	@abstractmethod
	def start_training(self) -> None:
		raise NotImplementedError(f"{self.__class__.__name__}: start_training method must be implemented in subclasses")
	
	@abstractmethod
	def generate_simulation_parameters(self, brain: Brain) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: generate_simulation_parameters method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")