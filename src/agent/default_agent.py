from src.agent			import Agent
from src.agent.brain	import Brain
from src.food			import Food

from math	import cos, sin, radians
from typing import Any


class DefaultAgent(Agent):
	def __init__(
			self, brain: Brain, width: int, height: int, lifespan: int, lifespan_extension: int,
			perception_distance: int, eating_distance: int
		):
		super().__init__(
			brain, width, height, lifespan, lifespan_extension, perception_distance,
			eating_distance, {"x": None, "y": None, "angle": None}
		)

	def set_history(self, history: list[tuple[Any]], keys: list[str] = ["x", "y", "angle"]) -> None:
		super().set_history(history, keys)

	def simulate(self, time_step: int, food_list: list[Food], agent_list: list[Agent]) -> None:
		if self.alive:
			if (time_step == self.lifespan):
				self.alive = False
				self.last_time_step = time_step
			else:
				l_rot, r_rot, speed = self.brain.get_action(self.state, self.perception_distance, food_list, agent_list)
				change = -3 if l_rot else 3 if r_rot else 0
				self.set_in_state("angle", self.get_from_state("angle") + change)
				self.set_in_state("angle", (self.get_from_state("angle") + 180) % 360 - 180)
				if speed:
					self.set_in_state("x", int((self.get_from_state("x") + cos(radians(self.get_from_state("angle"))) * 4) % self.width))
					self.set_in_state("y", int((self.get_from_state("y") + sin(radians(self.get_from_state("angle"))) * 4) % self.height))
				food, dist = self.brain.get_closest_food(self.state, food_list)
				if food != None and dist < self.eating_distance:
					food.eaten_by += [self]
			self.save_state()
	

	def to_dict(self) -> dict[str, Any]:
		return {
			"type"		: "default-agent",
			"lifetime"	: self.last_time_step
#             "history" : [(state["x"], state["y"], state["angle"]) for state in self.history]
		}

	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return (
			"brain", "width", "height", "agents-lifespan", "agents-lifespan-extension",
			"perception-distance", "eating-distance"
		)
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'DefaultAgent':
		for key in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
		return DefaultAgent(
			params["brain"], params["width"], params["height"], params["agents-lifespan"],
			params["agents-lifespan-extension"], params["perception-distance"], params["eating-distance"]
		)