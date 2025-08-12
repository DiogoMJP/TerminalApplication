from __future__	import annotations

from numpy import argmax

from src.agent.brain.perception_processors.perception_nodes	import PerceptionNode

from math	import acos, cos, degrees, hypot, sin, sqrt, radians
from typing	import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent								import Agent
	from src.food								import Food
	from src.agent.brain.perception_processors	import EnvironmentData


class EyesPerceptionNode(PerceptionNode):
	def __init__(
		self, n_cones: int, normalized: bool, fov: int, see_agents: bool,
		see_food: bool, see_poisonous_food: bool, see_walls: bool
	):
		super().__init__(
			n_cones * (0 if not see_food else 2 if see_poisonous_food else 1) +\
			n_cones * (1 if see_agents else 0) + n_cones * (1 if see_walls else 0)
		)
		self.n_cones			: int	= n_cones
		self.normalized			: bool	= normalized
		self.fov				: int	= fov
		self.see_agents			: bool	= see_agents
		self.see_food			: bool	= see_food
		self.see_poisonous_food	: bool	= see_poisonous_food
		self.see_walls			: bool	= see_walls

		self.view_cone	: float	= fov / n_cones

	def process_entity(
		self, x: float, y: float, e_x: float, e_y: float,
		perception_distance_sq: float, cones: list[tuple[float, float]],
		cos_threshold: float
	) -> tuple[int, float] | None:
		dx = e_x - x
		dy = e_y - y
		dist_sq = dx * dx + dy * dy
		if dist_sq == 0 or dist_sq > perception_distance_sq:
			return
		
		dist = sqrt(dist_sq)

		norm_dx = dx / dist
		norm_dy = dy / dist

		best_i = -1
		best_dot = cos_threshold
		for i, (cx, cy) in enumerate(cones):
			dot_val = norm_dx * cx + norm_dy * cy
			if dot_val > best_dot:
				best_dot = dot_val
				best_i = i
		
		return (best_i, dist) if best_i != -1 else None
	
	def process_agents(
		self, state: dict[str, Any], perception_distance: float,
		agent_list: list[Agent], cones: list[tuple[float, float]]
	) -> list[float]:
		x, y = state["x"], state["y"]
		output = [perception_distance] * self.n_cones
		perception_distance_sq = perception_distance ** 2
		cos_threshold = cos(radians(self.view_cone / 2))

		for agent in agent_list:
			result = self.process_entity(
				x, y, agent.get_from_state("x"), agent.get_from_state("y"),
				perception_distance_sq, cones, cos_threshold
			)
			if result:
				i, dist = result
				if dist < output[i]: output[i] = dist

		if self.normalized:
			return [1 - (val / perception_distance) for val in output]
		else:
			return [perception_distance - val for val in output]
	
	def process_food(
		self, state: dict[str, Any], perception_distance: float, poisonous_detection_distance: int,
		food_list: list[Food], cones: list[tuple[float, float]]
	) -> list[list[float]]:
		x, y = state["x"], state["y"]
		perception_distance_sq = perception_distance ** 2
		cos_threshold = cos(radians(self.view_cone / 2))

		regular_output = [perception_distance] * self.n_cones
		poisonous_output = [perception_distance] * self.n_cones
		
		for food in food_list:
			result = self.process_entity(
				x, y, food.x, food.y, perception_distance_sq,
				cones, cos_threshold
			)
			if result is None: continue
			i, dist = result
			if self.see_poisonous_food and food.poisonous and dist <= poisonous_detection_distance:
				if dist < poisonous_output[i]: poisonous_output[i] = dist
			else:
				if dist < regular_output[i]: regular_output[i] = dist
		
		def normalize(output):
			if not self.normalized:
				return [perception_distance - val for val in output]
			else:
				return [1 - (val / perception_distance) for val in output]
			
		if self.see_poisonous_food:
			return [normalize(regular_output), normalize(poisonous_output)]
		else:
			return [normalize(regular_output)]
	
	def process_walls(
		self, state: dict[str, Any], perception_distance: float, width: int, height: int,
		cones: list[tuple[float, float]]
	) -> list[float]:
		x, y = state["x"], state["y"]
		output = [0.0 for _ in range(self.n_cones)]

		for i, (dx, dy) in enumerate(cones):
			min_dist = perception_distance

			if dx != 0.0:
				t = (0 - x) / dx
				if t >= 0:
					y_hit = y + dy * t
					if 0 <= y_hit <= height: min_dist = min(min_dist, t)
				t = (width - x) / dx
				if t >= 0:
					y_hit = y + dy * t
					if 0 <= y_hit <= height: min_dist = min(min_dist, t)
			if dy != 0.0:
				t = (0 - y) / dy
				if t >= 0:
					x_hit = x + dx * t
					if 0 <= x_hit <= width: min_dist = min(min_dist, t)
				t = (height - y) / dy
				if t >= 0:
					x_hit = x + dx * t
					if 0 <= x_hit <= width: min_dist = min(min_dist, t)
			output[i] = 1 - (min_dist / perception_distance) if self.normalized else (perception_distance - min_dist)

		return output
	
	def process(
    	self, state: dict[str, Any], perception_distance: int, poisonous_detection_distance: int, 
		width: int, height: int, **environment_data: Unpack[EnvironmentData]
	) -> list[float]:
		angle = radians(state["angle"])
		half_fov = radians(self.fov / 2)
		cone_step = radians(self.view_cone)
		cones = [
			(cos(angle - half_fov + cone_step * (i + 0.5)), sin(angle - half_fov + cone_step * (i + 0.5)))
			for i in range(self.n_cones)
		]
		output = [[] for _ in range(self.n_cones)]

		if self.see_agents:
			if "agent_list" not in environment_data:
				raise Exception(f"{self.__class__.__name__}: Missing required environment data: agent_list")
			agent_list = environment_data["agent_list"]
			agents_output = self.process_agents(state, perception_distance, agent_list, cones)
			for i in range(self.n_cones):
				output[i] += [agents_output[i]]
		
		if self.see_food:
			if "food_list" not in environment_data:
				raise Exception(f"{self.__class__.__name__}: Missing required environment data: food_list")
			food_list = environment_data["food_list"]
			food_output = self.process_food(
				state, perception_distance, poisonous_detection_distance, food_list, cones
			)
			for i in range(self.n_cones):
				output[i] += [food_output[j][i] for j in range(len(food_output))]
		
		if self.see_walls:
			walls_output = self.process_walls(state, perception_distance, width, height, cones)
			for i in range(self.n_cones):
				output[i] += [walls_output[i]]

		final_output = []
		for freq in output:
			max_idx = argmax(freq)
			final_output += [freq[i] if i == max_idx else 0.0 for i in range(len(freq))]

		return final_output

	def to_dict(self) -> dict[str, Any]:
		return {"type"	: "eyes-perception-node"}
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (
			("n-cones", int), ("normalized", bool), ("fov", int), ("see-agents", bool),
			("see-food", bool), ("see-poisonous-food", bool), ("see-walls", bool)
		)
	
	@staticmethod
	def create_from_parameters(params) -> 'EyesPerceptionNode':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		return EyesPerceptionNode(
			params["n-cones"], params["normalized"], params["fov"], params["see-agents"],
			params["see-food"], params["see-poisonous-food"], params["see-walls"]
		)

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'EyesPerceptionNode':
		return EyesPerceptionNode(
			data["n-cones"], data["normalized"], data["fov"], data["see-agents"],
			data["see-food"], data["see-poisonous-food"], data["see-walls"]
		)