from __future__	import annotations

from src.utils												import CreatableFromParameters, Loadable
from src.agent.brain.perception_processors.perception_nodes	import SoundPerceptionNode, create_perception_node

from math	import sqrt
from typing	import Any, Optional, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors.perception_nodes	import PerceptionNode
	from src.agent.brain.perception_processors					import EnvironmentData
	from src.food												import Food


class PerceptionProcessor(CreatableFromParameters, Loadable):
	def __init__(
		self, perception_nodes: list[PerceptionNode]
	):
		self.perception_nodes	: list[PerceptionNode]	= perception_nodes

		self.n_cones			: Optional[int]			= None
		self.normalized			: Optional[bool]		= None
		self.fov				: Optional[int]			= None
		self.see_agents			: Optional[bool]		= None
		self.see_food			: Optional[bool]		= None
		self.see_poisonous_food	: Optional[bool]		= None
		self.see_walls			: Optional[bool]		= None
		self.n_freq				: Optional[int]			= None
	
	def get_n_input(self) -> int:
		return sum([perception_node.n_output for perception_node in self.perception_nodes])
	def get_n_output(self) -> int:
		if any([isinstance(pn, SoundPerceptionNode) for pn in self.perception_nodes]) and self.n_freq != None:
			return 3 + self.n_freq
		else:
			return 3
	
	def get_closest_food(self, state: dict[str, Any], food_list: list[Food]) -> tuple[Food | None, float]:
		x, y = state["x"], state["y"]

		min_dist_sq = float('inf')
		closest_food = None

		for food in food_list:
			if not food.alive:
				continue
			dx = x - food.x
			dy = y - food.y
			dist_sq = dx * dx + dy * dy
			if dist_sq < min_dist_sq:
				min_dist_sq = dist_sq
				closest_food = food

		if closest_food is None:
			return None, 0.0

		return closest_food, sqrt(min_dist_sq)

	def process_input(
    	self, state: dict[str, Any], perception_distance: int, poisonous_detection_distance: int, 
		width: int, height: int, **environment_data: Unpack[EnvironmentData]
	) -> tuple[float, ...]:
		output = []
		for node in self.perception_nodes:
			output += node.process(
				state, perception_distance, poisonous_detection_distance,
				width, height, **environment_data
			)
		return tuple(output)
	
	def to_dict(self) -> dict[str, Any]:
		data = {
			"type"					: "default-perception-processor",
			"perception-nodes"		: [node.to_dict() for node in self.perception_nodes]
		}
		if self.normalized != None: data |= {"normalized" : self.normalized}
		if self.n_cones != None: data |= {"n-cones" : self.n_cones}
		if self.fov != None: data |= {"fov" : self.fov}
		if self.see_agents != None: data |= {"see-agents" : self.see_agents}
		if self.see_food != None: data |= {"see-food" : self.see_food}
		if self.see_poisonous_food != None: data |= {"see-poisonous-food" : self.see_poisonous_food}
		if self.see_walls != None: data |= {"see-walls" : self.see_walls}
		if self.n_freq != None: data |= {"n-freq" : self.n_freq}
		
		return data
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (("perception-nodes", list),)
	
	@staticmethod
	def create_from_parameters(params) -> 'PerceptionProcessor':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		perception_nodes = [
			create_perception_node(node_type, params) for node_type in params["perception-nodes"]
		]
		perception_processor = PerceptionProcessor(perception_nodes)
		if "normalized" in params: perception_processor.normalized = params["normalized"]
		if "n-cones" in params: perception_processor.n_cones = params["n-cones"]
		if "fov" in params: perception_processor.fov = params["fov"]
		if "see-agents" in params: perception_processor.see_agents = params["see-agents"]
		if "see-food" in params: perception_processor.see_food = params["see-food"]
		if "see-poisonous-food" in params: perception_processor.see_poisonous_food = params["see-poisonous-food"]
		if "see-walls" in params: perception_processor.see_walls = params["see-walls"]
		if "n-freq" in params: perception_processor.n_freq = params["n-freq"]
		return perception_processor

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'PerceptionProcessor':
		perception_nodes = [
			create_perception_node(perception_node["type"], data) for perception_node in data["perception-nodes"]
		]
		return PerceptionProcessor(perception_nodes)