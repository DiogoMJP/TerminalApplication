from __future__	import annotations

from src.agent.brain	import Brain
from src.simulation 	import Simulation

from random import random
from typing import Any


class RandomFoodSimulation(Simulation):
	def __init__(
		self, brain : Brain, width: int, height: int, n_agents: int, agent_type: str, agents_lifespan: int,
		agents_lifespan_extension: int, food_type: str, food_lifespan: int, perception_distance: int,
		eating_distance: int, eating_number: int, max_time_steps: int, food_spawn_rate: float
	):
		super().__init__(
			brain, width, height, n_agents, agent_type, agents_lifespan, agents_lifespan_extension,
			food_type, food_lifespan, 0.0, 0, perception_distance, eating_distance, eating_number,
			max_time_steps
		)
		self.food_spawn_rate	: float	= food_spawn_rate
		
		self.create_agents()
		self.start_loop()
	
	def main_loop(self) -> None:
		while not self.finished:
			for food in self.food:
				food.simulate(self.time_step)
			self.separate_food()
			if random() < self.food_spawn_rate:
				self.create_food()
			for i, agent in enumerate(self.agents):
				agent.simulate(self.time_step, self, self.agents[:i]+self.agents[i+1:])
			if self.time_step == self.max_time_steps - 1 or self.get_n_alive_agents() == 0:
				self.finished = True
				self.last_time_step = self.time_step
				break
			self.update_sound_history()
			self.sounds = []
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
			"type"		: "random-food-simulation",
			"duration"	: self.last_time_step,
			"brain"		: self.brain.to_dict(),
			"agents"	: [agent.to_dict() for agent in self.agents],
			"food"		: [food.to_dict() for food in self.finished_food]
		}
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (
			("brain", Brain), ("width", int), ("height", int), ("n-agents", int), ("agent-type", str),
			("agents-lifespan", int), ("agents-lifespan-extension", int), ("food-type", str), ("food-lifespan", int),
			("perception-distance", int), ("eating-distance", int), ("eating-number", int), ("max-time-steps", int),
			("food-spawn-rate", float)
		)
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'RandomFoodSimulation':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		return RandomFoodSimulation(
			params["brain"], params["width"], params["height"], params["n-agents"], params["agent-type"],
			params["agents-lifespan"], params["agents-lifespan-extension"], params["food-type"], params["food-lifespan"],
			params["perception-distance"], params["eating-distance"], params["eating-number"], params["max-time-steps"],
			params["food-spawn-rate"]
		)