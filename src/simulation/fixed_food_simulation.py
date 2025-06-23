from __future__	import annotations

from src.simulation	import Simulation

from time		import time
from typing 	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain


class FixedFoodSimulation(Simulation):
	def __init__(
		self, brain : Brain, width: int, height: int, n_agents: int, agent_type: str, agents_lifespan: int,
		agents_lifespan_extension: int, food_type: str, food_lifespan: int, perception_distance: int,
		eating_distance: int, eating_number: int, max_time_steps: int, n_food: int
	):
		super().__init__(
			brain, width, height, n_agents, agent_type, agents_lifespan, agents_lifespan_extension,
			food_type, food_lifespan, perception_distance, eating_distance, eating_number, max_time_steps
		)
		self.n_food	: int	= n_food
		self.create_agents()
		while self.get_n_food() < self.n_food:
			self.create_food()
		self.start_loop()
	
	def main_loop(self) -> None:
		while not self.finished:
			for food in self.food:
				food.simulate(self.time_step)
			self.separate_food()
			while self.get_n_food() < self.n_food:
				self.create_food()
			for i, agent in enumerate(self.agents):
				agent.simulate(self.time_step, self.food, self.agents[:i]+self.agents[i+1:])
			if self.time_step == self.max_time_steps - 1 or self.get_n_alive_agents() == 0:
				self.finished = not self.finished
				self.last_time_step = self.time_step
				break
			self.time_step += 1
		for agent in self.agents:
			if agent.alive == True:
				agent.alive = False
				agent.last_time_step = self.time_step
		for food in self.food:
			if food.alive == True:
				food.alive = False
				food.last_time_step = self.time_step
			self.finished_food += [food]
	
	def to_dict(self) -> dict[str, Any]:
		return {
			"type"		: "fixed-food-simulation",
			"duration"	: self.last_time_step,
			"brain"		: self.brain.to_dict(),
			"agents"	: [agent.to_dict() for agent in self.agents],
			"food"		: [food.to_dict() for food in self.finished_food]
		}
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return (
			"brain", "width", "height", "n-agents", "agent-type", "agents-lifespan", "agents-lifespan-extension",
			"food-type", "food-lifespan", "perception-distance", "eating-distance", "eating-number", "max-time-steps",
			"n-food"
		)
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'FixedFoodSimulation':
		for key in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
		return FixedFoodSimulation(
			params["brain"], params["width"], params["height"], params["n-agents"], params["agent-type"],
			params["agents-lifespan"], params["agents-lifespan-extension"], params["food-type"], params["food-lifespan"],
			params["perception-distance"], params["eating-distance"], params["eating-number"],
			params["max-time-steps"], params["n-food"]
		)