from __future__	import annotations

from src.utils	import CreatableFromParameters

from abc				import abstractmethod
from typing				import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain
	from src.simulation		import Simulation


class Agent(CreatableFromParameters):
	def __init__(
		self, id: int, brain: Brain, width: int, height: int, lifespan: int, lifespan_extension: int,
		perception_distance: int, poisonous_perception_distance: int,eating_distance: int,
		state: dict[str, Any]
	):
		self.id								: int				= id
		self.brain							: Brain				= brain
		self.width							: int				= width
		self.height							: int				= height
		self.lifespan						: int				= lifespan
		self.lifespan_extension				: int				= lifespan_extension
		self.perception_distance			: int				= perception_distance
		self.poisonous_perception_distance	: int				= poisonous_perception_distance
		self.eating_distance				: int				= eating_distance
		self.state							: dict[str, Any]	= state

		self.last_time_step	: int					= 0
		self.history		: list[dict[str, Any]]	= []
		self.alive			: bool					= True

	def get_from_state(self, key: str) -> Any:
		return self.state[key]
	def set_in_state(self, key: str, value: Any) -> None:
		self.state[key] = value
	def get_from_history(self, time_step: int, key: str) -> Any:
		return self.history[time_step][key]
	def prolong_lifespan(self, time_step: int) -> None:
		self.lifespan = time_step + self.lifespan_extension
	def end_lifespan(self, time_step: int) -> None:
		self.alive = False
		self.last_time_step = time_step
	
	def set_history(self, history: list[tuple[Any]] , keys: list[str]) -> None:
		for state in history:
			self.history += [{keys[i]: val for i, val in enumerate(state)}]
	def save_state(self) -> None:
		self.history += [{key: val for key, val in self.state.items()}]

	@abstractmethod
	def simulate(self, time_step: int, simulation: Simulation, agent_list: list[Agent]) -> None:
		raise NotImplementedError(f"{self.__class__.__name__}: simulate method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")