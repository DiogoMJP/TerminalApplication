from __future__ import annotations

from src.simulation	import get_simulation_parameters
from src.utils		import Loadable

from abc	import abstractmethod
from typing	import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain


class TrainingReplay(Loadable):
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
		self.food_spawn_rate			: Optional[float]	= None
		self.n_food						: Optional[float]	= None
		self.brain						: Optional[Brain]	= None
		self.n_sensors					: Optional[int]		= None
		self.fov						: Optional[int]		= None
	
	def generate_simulation_parameters(self, brain: Brain) -> dict[str, Any]:
		simulation_params = get_simulation_parameters(self.simulation_type)
		params = {
			"brain"						: brain,
			"width"						: self.width,
			"height"					: self.height,
			"n-agents"					: self.n_agents,
			"agent-type"				: self.agent_type,
			"agents-lifespan"			: self.agents_lifespan,
			"agents-lifespan-extension"	: self.agents_lifespan_extension,
			"food-lifespan"				: self.food_lifespan,
			"perception-distance"		: self.perception_distance,
			"eating-distance"			: self.eating_distance,
			"eating-number"				: self.eating_number,
			"max-time-steps"			: self.max_time_steps
		}
		if "food-spawn-rate" in simulation_params and self.food_spawn_rate != None:
			params["food-spawn-rate"] = self.food_spawn_rate
		if "n-food" in simulation_params and self.n_food != None:
			params["n-food"] = self.n_food
		return params
	
	@abstractmethod
	def create_graphs(self, path: str) -> None:
		raise NotImplementedError(f"{self.__class__.__name__}: create_graphs method must be implemented in subclasses")