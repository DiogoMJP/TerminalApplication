from __future__	import annotations

from src.agent.brain.perception_processors	import get_perception_processor_parameters
from src.utils	import CreatableFromParameters

from abc	import abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain


class Training(CreatableFromParameters):
	def __init__(
		self, n_generations: int, width: int, height: int, n_agents: int, agent_type: str, agents_lifespan: int,
		agents_lifespan_extension: int, food_lifespan: int, perception_distance: int, eating_distance: int,
		eating_number: int, max_time_steps: int, perception_processor_type: str, simulation_type: str
	):
		self.n_generations				: int				= n_generations
		self.width						: int				= width
		self.height						: int				= height
		self.n_agents					: int				= n_agents
		self.agent_type					: str				= agent_type
		self.agents_lifespan			: int				= agents_lifespan
		self.agents_lifespan_extension	: int				= agents_lifespan_extension
		self.food_lifespan				: int				= food_lifespan
		self.perception_distance		: int				= perception_distance
		self.eating_distance			: int				= eating_distance
		self.eating_number				: int				= eating_number
		self.max_time_steps				: int				= max_time_steps
		self.perception_processor_type	: str				= perception_processor_type
		self.simulation_type			: str				= simulation_type
		self.brain						: Optional[Brain]	= None
		self.n_sensors					: Optional[int]		= None
		self.fov						: Optional[int]		= None
	
	@abstractmethod
	def start_training(self) -> None:
		raise NotImplementedError(f"{self.__class__.__name__}: start_training method must be implemented in subclasses")
	
	def generate_perception_processor_parameter(self) -> dict[str, Any]:
		perception_processor_params = get_perception_processor_parameters(self.perception_processor_type)
		params = {}
		if "n-sensors" in perception_processor_params and self.n_sensors != None:
			params["n-sensors"] = self.n_sensors
		if "fov" in perception_processor_params and self.fov != None:
			params["fov"] = self.fov
		return params
	
	@abstractmethod
	def generate_simulation_parameters(self, brain: Brain) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: generate_simulation_parameters method must be implemented in subclasses")
	
	def to_dict(self) -> dict[str, Any]:
		data = {
			"n-generations" : self.n_generations,
			"width" : self.width,
			"height" : self.height,
			"n-agents" : self.n_agents,
			"agents-lifespan" : self.agents_lifespan,
			"agents-lifespan-extension" : self.agents_lifespan_extension,
			"food-lifespan" : self.food_lifespan,
			"perception-distance" : self.perception_distance,
			"eating-distance" : self.eating_distance,
			"eating-number" : self.eating_number,
			"max-time-steps" : self.max_time_steps,
			"simulation-type" : self.simulation_type
        }
		if self.brain != None: data |= {"brain" : self.brain.to_dict()}
		if self.n_sensors != None: data |= {"n-sensors" : self.n_sensors}
		if self.fov != None: data |= {"fov" : self.fov}
		return data