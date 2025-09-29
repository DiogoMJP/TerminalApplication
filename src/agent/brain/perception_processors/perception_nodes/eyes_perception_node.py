from __future__	import annotations

from src.agent.brain.perception_processors.perception_nodes	import PerceptionNode

import numpy	as np
from math	import cos, radians, sin, sqrt
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
	
	def _normalize(self, values: np.ndarray, perception_distance: float) -> np.ndarray:
		if self.normalized:
			return 1.0 - (values / perception_distance)
		return perception_distance - values
	
	def process_entity_batch(
		self, x: float, y: float, entities: np.ndarray,
		perception_distance_sq: float, cones: np.ndarray, cos_threshold: float
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		if entities.size == 0:
			return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)
		
		dxdy_full = entities - np.array([x, y])
		dist_sq_full = np.sum(dxdy_full**2, axis=1)

		# filter out too far or overlapping entities
		valid_mask = (dist_sq_full > 0) & (dist_sq_full <= perception_distance_sq)
		if not np.any(valid_mask):
			return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)
		
		valid_indices = np.nonzero(valid_mask)[0]
		dxdy = dxdy_full[valid_mask]
		dist = np.sqrt(dist_sq_full[valid_mask])[:, None]
		normed = dxdy / dist

		# dot products with all cones → shape (N, n_cones)
		dots = normed @ cones.T

		# best cone per entity
		best_idx = np.argmax(dots, axis=1)
		best_val = np.max(dots, axis=1)

		# keep only those inside FOV
		keep_mask = best_val > cos_threshold
		if not np.any(keep_mask):
			return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)
		
		kept_cone_indices = best_idx[keep_mask]
		kept_dists = dist.flatten()[keep_mask]
		kept_original_indices = valid_indices[keep_mask]

		return kept_cone_indices, kept_dists, kept_original_indices

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
	
	def process_agents(self, state: dict[str, Any], perception_distance: float,
		agent_list: list[Agent], cones: np.ndarray
	) -> np.ndarray:
		if not agent_list:
			return np.zeros(self.n_cones)

		entities = np.array([[a.get_from_state("x"), a.get_from_state("y")] for a in agent_list])
		indices, dists, _kept  = self.process_entity_batch(
			state["x"], state["y"], entities,
			perception_distance**2, cones, cos(radians(self.view_cone / 2))
		)

		output = np.full(self.n_cones, perception_distance, dtype=float)
		if indices.size:
			np.minimum.at(output, indices, dists)  # inplace min update
		return self._normalize(output, perception_distance)
	
	def process_food(self, state: dict[str, Any], perception_distance: float,
		poisonous_detection_distance: int, food_list: list[Food], cones: np.ndarray
	) -> list[np.ndarray]:
		if not food_list:
			if self.see_poisonous_food:
				return [np.zeros(self.n_cones), np.zeros(self.n_cones)]
			return [np.zeros(self.n_cones)]

		entities = np.array([[f.x, f.y] for f in food_list])
		indices, dists, kept_original_indices = self.process_entity_batch(
			state["x"], state["y"], entities,
			perception_distance**2, cones, cos(radians(self.view_cone / 2))
		)

		regular = np.full(self.n_cones, perception_distance, dtype=float)
		poisonous = np.full(self.n_cones, perception_distance, dtype=float)

		if indices.size:
			if self.see_poisonous_food:
				# produce boolean array of poisonous flags for kept items
				poison_flags = np.array([food_list[i].poisonous for i in kept_original_indices], dtype=bool)
				# mask where a kept item is considered poisonous (and within poisonous_detection_distance)
				poison_by_dist = (dists <= poisonous_detection_distance) & poison_flags

				# update poisonous channel for those that are poisonous & within poison detection distance
				if np.any(poison_by_dist):
					np.minimum.at(poisonous, indices[poison_by_dist], dists[poison_by_dist])

				# the remaining items update the regular channel
				regular_mask = ~poison_by_dist
				if np.any(regular_mask):
					np.minimum.at(regular, indices[regular_mask], dists[regular_mask])
			else:
				# all detected food goes to regular channel
				np.minimum.at(regular, indices, dists)

		if self.see_poisonous_food:
			return [self._normalize(regular, perception_distance),
					self._normalize(poisonous, perception_distance)]
		return [self._normalize(regular, perception_distance)]
	
	def process_walls(self, state: dict[str, Any], perception_distance: float,
		width: int, height: int, cones: np.ndarray
	) -> np.ndarray:
		x, y = state["x"], state["y"]
		output = np.empty(self.n_cones, dtype=float)

		for i, (dx, dy) in enumerate(cones):
			min_dist = perception_distance
			if dx != 0.0:
				for bx in (0, width):
					t = (bx - x) / dx
					if t >= 0:
						y_hit = y + dy * t
						if 0 <= y_hit <= height: min_dist = min(min_dist, t)
			if dy != 0.0:
				for by in (0, height):
					t = (by - y) / dy
					if t >= 0:
						x_hit = x + dx * t
						if 0 <= x_hit <= width: min_dist = min(min_dist, t)

			output[i] = min_dist

		return self._normalize(output, perception_distance)
	
	def process(
    	self, state: dict[str, Any], perception_distance: int, poisonous_detection_distance: int, 
		width: int, height: int, **environment_data: Unpack[EnvironmentData]
	) -> list[float]:
		angle = radians(state["angle"])
		half_fov = radians(self.fov / 2)
		cone_step = radians(self.view_cone)

		cones = np.array([
			(cos(angle - half_fov + cone_step * (i + 0.5)),
			 sin(angle - half_fov + cone_step * (i + 0.5)))
			for i in range(self.n_cones)
		])

		output_parts = []

		if self.see_agents:
			agents_output = self.process_agents(state, perception_distance,
				environment_data.get("agent_list", []), cones)
			output_parts.append(agents_output)

		if self.see_food:
			food_output = self.process_food(state, perception_distance,
				poisonous_detection_distance, environment_data.get("food_list", []), cones)
			output_parts.extend(food_output)

		if self.see_walls:
			walls_output = self.process_walls(state, perception_distance, width, height, cones)
			output_parts.append(walls_output)

		if not output_parts:
			return [0.0] * (self.n_cones * self.n_output)

		stacked = np.stack(output_parts, axis=1)  # (n_cones, features)
		max_idx = np.argmax(stacked, axis=1)
		final_output = np.zeros_like(stacked)

		for i, j in enumerate(max_idx):
			final_output[i, j] = stacked[i, j]

		return final_output.flatten().tolist()

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