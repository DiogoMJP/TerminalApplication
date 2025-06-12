from __future__	import annotations

from src.agent.brain						import Brain
from src.agent.brain.neural_network			import NeatNeuralNetwork, create_neural_network
from src.agent.brain.perception_processors	import PerceptionProcessor, \
	create_perception_processor, load_perception_processor_from_data

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent	import Agent
	from src.food	import Food


class NeatBrain(Brain):
	def __init__(self, neat_net: NeatNeuralNetwork, perception_processor: PerceptionProcessor):
		super().__init__(3, perception_processor)
		self.neat_net	: NeatNeuralNetwork	= neat_net
	
	def get_action(
		self, state: dict[str, Any], perception_distance: int, food_list: list[Food], agent_list: list[Agent]
	) -> tuple[int, int, int, float, Food|None]:
		input = self.perception_processor.process_input(state, perception_distance, food_list, agent_list)
		
		food = input[0]
		f_dist = input[2]
		input = input[1:]
		
		output = self.neat_net.activate(input)
		l_rot = 1 if output[0] >= 0.5 else 0
		r_rot = 1 if output[1] >= 0.5 else 0
		speed = 1 if output[2] >= 0.5 else 0
		
		return (l_rot, r_rot, speed, f_dist, food)
	

	def to_dict(self) -> dict[str, Any]:
		return {
			"type" : "neat-brain",
			"network" : self.neat_net.to_dict(),
			"perception-processor" : self.perception_processor.to_dict()
		}
	
	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'NeatBrain':
		return NeatBrain(
			NeatNeuralNetwork.load_from_data(data["network"]),
			load_perception_processor_from_data(data["perception-processor"])
		)
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return ("neat-neural-network", "perception-processor")
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'NeatBrain':
		for key in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
		try:
			neural_network = create_neural_network("neat-neural-network", params)
			perception_processor = create_perception_processor(params["perception-processor-type"], params)
		except Exception as e: raise
		return NeatBrain(neural_network, perception_processor)