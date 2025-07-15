from __future__	import annotations

from src.agent.brain.perception_processors	import PerceptionProcessor

from math			import acos, cos, degrees, radians, sin, sqrt
from typing			import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import EnvironmentData


class PoisonousEyesPerceptionProcessor(PerceptionProcessor):
	def __init__(self, n_sensors: int, fov: int):
		super().__init__(n_sensors * 3)
		self.n_sensors	: int	= n_sensors
		self.fov		: int	= fov
		self.view_cone	: float	= fov / n_sensors
	
	def process_input(
    	self, state: dict[str, Any], perception_distance: int, width: int, height: int,
		**environment_data: Unpack[EnvironmentData]
	) -> tuple[int | float, ...]:
		if "food_list" not in environment_data:
			raise Exception(f"{self.__class__.__name__}: Missing required environment data: food_list")
		if "agent_list" not in environment_data:
			raise Exception(f"{self.__class__.__name__}: Missing required environment data: agent_list")
		food_list = environment_data["food_list"]
		agent_list = environment_data["agent_list"]
		
		x, y = state["x"], state["y"]
		
		angle = state["angle"]
		cones = [
			(cos(r), sin(r)) for r in (
				radians(angle - self.fov / 2 + self.view_cone / 2 + i * self.view_cone)
				for i in range(self.n_sensors)
			)
		]

		output = [[0.0, 0.0, 0.0] for _ in range(self.n_sensors)]

		def process_entity(e_x: float, e_y: float, entity_key: int) -> None:
			dx = e_x - x
			dy = e_y - y
			dist_sq = dx * dx + dy * dy
			if dist_sq == 0:
				return
			dist = sqrt(dist_sq)
			if dist > perception_distance:
				return

			norm_dx = dx / dist
			norm_dy = dy / dist

			best_i = -1
			best_dot = -1.0
			for i, (cx, cy) in enumerate(cones):
				dot_val = norm_dx * cx + norm_dy * cy
				if dot_val > best_dot:
					best_dot = dot_val
					best_i = i
			if degrees(acos(max(min(best_dot, 1), 0))) < self.view_cone / 2:
				rel_dist = dist / perception_distance
				if 1 - max(output[best_i]) > rel_dist:
					output[best_i] = [0.0, 0.0, 0.0]
					output[best_i][entity_key] = 1 - rel_dist

		for food in food_list:
			if food.alive:
				process_entity(food.x, food.y, 1 if food.poisonous else 0)

		for agent in agent_list:
			if agent.alive:
				process_entity(agent.get_from_state("x"), agent.get_from_state("y"), 2)
		
		# for i in range(self.n_sensors):
		# 	if output[i][1] == 0.0 and output[i][2] == 0.0:
		# 		if x + perception_distance * cones[i][0] < 0 or x + perception_distance * cones[i][0] >= width or \
		# 			y + perception_distance * cones[i][1] < 0 or y + perception_distance * cones[i][1] >= height:
		# 			output[i] = [1.0, 0.0, 0.0, 0.0]
		
		return tuple(val for set in output for val in set)

	def to_dict(self) -> dict[str, Any]:
		return {
			"type"		: "poisonous-eyes-perception-processor",
			"n-sensors"	: self.n_sensors,
			"fov"		: self.fov
		}
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (("n-sensors", int), ("fov", int))
	
	@staticmethod
	def create_from_parameters(params) -> 'PoisonousEyesPerceptionProcessor':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		return PoisonousEyesPerceptionProcessor(params["n-sensors"], params["fov"])

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'PoisonousEyesPerceptionProcessor':
		return PoisonousEyesPerceptionProcessor(data["n-sensors"], data["fov"])