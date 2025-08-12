from __future__	import annotations

from src.agent.brain.perception_processors.perception_nodes	import PerceptionNode

from math	import atan2, degrees, hypot
from typing	import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import EnvironmentData
	from src.agent								import Agent
	from src.food								import Food


class AngleDistancePerceptionNode(PerceptionNode):
	def __init__(
		self, normalized: bool, see_agents: bool, see_food: bool, see_poisonous_food: bool
	):
		super().__init__(
			(2 if see_agents else 0) +\
			(0 if not see_food else 3 if see_poisonous_food else 2)
		)
		self.normalized			: bool	= normalized
		self.see_agents			: bool	= see_agents
		self.see_food			: bool	= see_food
		self.see_poisonous_food	: bool	= see_poisonous_food

	def process_agents(
		self, state: dict[str, Any], perception_distance: int, agent_list: list[Agent]
	) -> list[float]:
		x = state["x"]; y = state["y"]; angle = state["angle"]
		output = [0.0, perception_distance]
		
		for agent in agent_list:
			if agent.alive:
				dx = agent.get_from_state("x") - x; dy = agent.get_from_state("y") - y
				dist = hypot(dx, dy)
				if dist <= perception_distance and dist < output[1]:
					output[0] = (degrees(atan2(dy, dx)) - angle + 180) % 360 - 180
					output[1] = dist
		
		if self.normalized:
			return [output[0] / 180, 1 - (output[1] / perception_distance)]
		else:
			return [output[0], perception_distance - output[1]]
	
	def process_food(
		self, state: dict[str, Any], perception_distance: int, 
		poisonous_detection_distance: int, food_list: list[Food]
	) -> list[float]:
		x = state["x"]; y = state["y"]; angle = state["angle"]
		output = [0.0, perception_distance]; poisonous = 0.0
		
		for food in food_list:
			if food.alive:
				dx = food.x - x; dy = food.y - y
				dist = hypot(dx, dy)
				if dist <= perception_distance and dist < output[1]:
					output[0] = (degrees(atan2(dy, dx)) - angle + 180) % 360 - 180
					output[1] = dist
					if self.see_poisonous_food:
						if poisonous_detection_distance < dist:
							poisonous = 0.0
						else:
							poisonous = -1.0 if food.poisonous else 1.0
		
		if self.normalized:
			output = [output[0] / 180, 1 - (output[1] / perception_distance)]
		else:
			output = [output[0], perception_distance - output[1]]
		
		if self.see_poisonous_food:
			return [output[0], output[1], poisonous]
		else:
			return [output[0], output[1]]

	def process(
    	self, state: dict[str, Any], perception_distance: int, poisonous_detection_distance: int, 
		width: int, height: int, **environment_data: Unpack[EnvironmentData]
	) -> list[float]:
		output = []

		if self.see_agents:
			if "agent_list" not in environment_data:
				raise Exception(f"{self.__class__.__name__}: Missing required environment data: food_list")
			agent_list = environment_data["agent_list"]
			output += self.process_agents(state, perception_distance, agent_list)
		
		if self.see_food:
			if "food_list" not in environment_data:
				raise Exception(f"{self.__class__.__name__}: Missing required environment data: food_list")
			food_list = environment_data["food_list"]
			output += self.process_food(
				state, perception_distance, poisonous_detection_distance, food_list
			)
		
		return output

	def to_dict(self) -> dict[str, Any]:
		return {"type"	: "angle-distance-perception-node"}
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (
		("normalized", bool), ("see-agents", bool), ("see-food", bool),
		("see-poisonous-food", bool)
	)
	
	@staticmethod
	def create_from_parameters(params) -> 'AngleDistancePerceptionNode':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		return AngleDistancePerceptionNode(
			params["normalized"], params["see-agents"], params["see-food"],
			params["see-poisonous-food"]
		)

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'AngleDistancePerceptionNode':
		return AngleDistancePerceptionNode(
			data["normalized"], data["see-agents"], data["see-food"],
			data["see-poisonous-food"]
		)