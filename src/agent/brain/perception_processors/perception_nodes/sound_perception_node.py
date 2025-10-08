from __future__	import annotations

from src.agent.brain.perception_processors.perception_nodes	import PerceptionNode

from math	import atan2, degrees, tanh
from typing	import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import EnvironmentData


class SoundPerceptionNode(PerceptionNode):
	def __init__(self, n_freq: int):
		super().__init__(n_freq + 1)
		self.n_freq	: int	= n_freq
	
	def process(
    	self, state: dict[str, Any], perception_distance: int, poisonous_detection_distance: int, 
		width: int, height: int, **environment_data: Unpack[EnvironmentData]
	) -> list[float]:
		if "sound_list" not in environment_data:
			raise Exception(f"{self.__class__.__name__}: Missing required environment data: sound_list")
		sound_list = environment_data["sound_list"]

		x, y = state["x"], state["y"]; angle = state["angle"]

		ears_output = [0.0 for _ in range(self.n_freq)]
		sound_angle = 0.0

		if len(sound_list) != 0:
			sound_vec = [0.0, 0.0]
			for _, sound_pos, sound_freq in sound_list:
				dx = 0.04*(sound_pos[0] - x)
				dy = 0.04*(sound_pos[1] - y)
				dist_sq = dx*dx + dy*dy
				if dist_sq > 0:
					for freq, amplitude in enumerate(sound_freq):
						ears_output[freq] += amplitude / dist_sq
					if any(sound_freq) > 0:
						sound_vec[0] += dx / dist_sq
						sound_vec[1] += dy / dist_sq
			sound_angle = ((degrees(atan2(sound_vec[1], sound_vec[0])) - angle + 180) % 360 - 180) / 180
		
		return [tanh(val) for val in ears_output] + [sound_angle]

	def to_dict(self) -> dict[str, Any]:
		return {"type"	: "sound-perception-node"}
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (("n-freq", int),)
	
	@staticmethod
	def create_from_parameters(params) -> 'SoundPerceptionNode':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		return SoundPerceptionNode(params["n-freq"])

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'SoundPerceptionNode':
		return SoundPerceptionNode(data["n-freq"])