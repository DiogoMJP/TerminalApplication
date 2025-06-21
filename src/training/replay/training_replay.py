from __future__ import annotations

from src.utils	import Loadable

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