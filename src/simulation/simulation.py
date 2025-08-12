from __future__	import annotations

from src.agent	import create_agent, get_agent_parameters
from src.food	import create_food, get_food_parameters
from src.utils	import CreatableFromParameters

from abc		import abstractmethod
from random		import random
from threading	import Thread
from typing		import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent								import Agent
	from src.agent.brain						import Brain
	from src.agent.brain.perception_processors	import Sound
	from src.food								import Food


class Simulation(CreatableFromParameters):
	def __init__(
		self, brain : Brain, width: int, height: int, n_agents: int, agent_type: str, agents_lifespan: int,
		agents_lifespan_extension: int, food_type: str, food_lifespan: int, poisonous_food_rate: float,
		poisonous_perception_distance: int, perception_distance: int, eating_distance: int,
		eating_number: int, max_time_steps: int
	):
		self.brain							: Brain	= brain
		self.width							: int	= width
		self.height							: int	= height
		self.n_agents						: int	= n_agents
		self.agent_type						: str	= agent_type
		self.agents_lifespan				: int	= agents_lifespan
		self.agents_lifespan_extension		: int	= agents_lifespan_extension
		self.food_type						: str	= food_type
		self.food_lifespan					: int	= food_lifespan
		self.poisonous_food_rate			: float	= poisonous_food_rate
		self.perception_distance			: int	= perception_distance
		self.poisonous_perception_distance	: int	= poisonous_perception_distance
		self.eating_distance				: int	= eating_distance
		self.eating_number					: int	= eating_number
		self.max_time_steps					: int	= max_time_steps
		
		self.last_time_step		: int				= 0
		self.finished			: bool				= False
		self.time_step			: int				= 0
		self.agents				: list[Agent]		= []
		self.food				: list[Food]		= []
		self.finished_food		: list[Food]		= []
		self.sounds				: list[Sound]		= []
		self.main_loop_thread	: Optional[Thread]	= None

	def get_n_alive_agents(self) -> int:
		return len([1 for agent in self.agents if agent.alive])
	def get_n_food(self) -> int:
		return len([1 for food in self.food if food.alive])
	def get_n_eaten_food(self) -> int:
		return len([1 for food in self.finished_food if food.eaten])

	def create_agents(self) -> None:
		for _ in range(self.n_agents):
			params = self.generate_agent_parameters()
			agent = create_agent(self.agent_type, params)
			agent.set_in_state("x", int(random()*self.width))
			agent.set_in_state("y", int(random()*self.height))
			agent.set_in_state("angle", int(random()*360))
			agent.save_state()
			self.agents += [agent]
	def create_food(self) -> None:
		params = self.generate_food_parameters()
		self.food += [create_food(self.food_type, params)]
	def separate_food(self) -> None:
		dead_food = [food for food in self.food if not food.alive]
		self.finished_food += dead_food
		self.food = [food for food in self.food if food.alive]
	def add_sound(self, sound: Sound):
		self.sounds += [sound]
	
	def start_loop(self) -> None:
		if not self.finished:
			self.main_loop_thread = Thread(target=self.main_loop, name="name", args=[])
			self.main_loop_thread.start()
	
	def generate_agent_parameters(self) -> dict[str, Any]:
		agent_params = [param[0] for param in get_agent_parameters(self.agent_type)]
		params = {}
		if "brain" in agent_params:
			params["brain"] = self.brain
		if "width" in agent_params:
			params["width"] = self.width
		if "height" in agent_params:
			params["height"] = self.height
		if "agents-lifespan" in agent_params:
			params["agents-lifespan"] = self.agents_lifespan
		if "agents-lifespan-extension" in agent_params:
			params["agents-lifespan-extension"] = self.agents_lifespan_extension
		if "perception-distance" in agent_params:
			params["perception-distance"] = self.perception_distance
		if "poisonous-perception-distance" in agent_params:
			params["poisonous-perception-distance"] = self.poisonous_perception_distance
		if "eating-distance" in agent_params:
			params["eating-distance"] = self.eating_distance
		return params
	
	def generate_food_parameters(self) -> dict[str, Any]:
		food_params = [param[0] for param in get_food_parameters("default-food")]
		params = {}
		if "x" in food_params:
			params["x"] = int(random() * self.width)
		if "y" in food_params:
			params["y"] = int(random() * self.height)
		if "eating-number" in food_params:
			params["eating-number"] = self.eating_number
		if "first-time-step" in food_params:
			params["first-time-step"] = self.time_step
		if "food-lifespan" in food_params:
			params["food-lifespan"] = self.food_lifespan
		if "perception-distance" in food_params:
			params["perception-distance"] = self.perception_distance
		if "poisonous-perception-distance" in food_params:
			params["poisonous-perception-distance"] = self.poisonous_perception_distance
		if "poisonous-food-rate" in food_params:
			params["poisonous-food-rate"] = self.poisonous_food_rate
		return params

	@abstractmethod
	def main_loop(self) -> None:
		raise NotImplementedError(f"{self.__class__.__name__}: main_loop method must be implemented in subclasses")
	
	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		raise NotImplementedError(f"{self.__class__.__name__}: to_dict method must be implemented in subclasses")