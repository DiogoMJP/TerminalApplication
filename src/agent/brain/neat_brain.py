from __future__	import annotations

from src.agent.brain						import Brain
from src.agent.brain.neural_network			import NeatNeuralNetwork, create_neural_network
from src.agent.brain.perception_processors	import create_perception_processor, load_perception_processor_from_data

from neat.nn			import FeedForwardNetwork
from typing				import Any, Unpack, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.neural_network			import NeuralNetwork
	from src.agent.brain.perception_processors	import EnvironmentData, PerceptionProcessor


class NeatBrain(Brain):
	def __init__(self, neat_net: NeuralNetwork, perception_processor: PerceptionProcessor):
		if not isinstance(neat_net, NeatNeuralNetwork):
			raise Exception(f"{self.__class__.__name__}: Wrong neural network type: {type(neat_net)}")
		super().__init__(len(neat_net.outputs), perception_processor)
		self.neat_net	: NeatNeuralNetwork	= neat_net
	
	def get_n_nodes(self) -> int:
		return len(self.neat_net.node_evals) + len(self.neat_net.inputs)
	
	def get_action(
		self, state: dict[str, Any], perception_distance: int, width: int, height: int,
		**environment_data: Unpack[EnvironmentData]
	) -> tuple[int, ...]:
		input = self.get_perception(
			state, perception_distance, width, height, **environment_data
		)
		
		output = self.neat_net.activate(input)
		
		return tuple(1 if val >= 0.5 else 0 for val in output)

	def to_dict(self) -> dict[str, Any]:
		return {
			"type"					: "neat-brain",
			"network"				: self.neat_net.to_dict(),
			"perception-processor"	: self.perception_processor.to_dict()
		}
	
	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'NeatBrain':
		return NeatBrain(
			NeatNeuralNetwork.load_from_data(data["network"]),
			load_perception_processor_from_data(data["perception-processor"])
		)
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (("neat-neural-network", FeedForwardNetwork), ("perception-processor-type", str))
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'NeatBrain':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		try:
			neural_network = create_neural_network("neat-neural-network", params)
			perception_processor = create_perception_processor(params["perception-processor-type"], params)
		except Exception as e: raise
		return NeatBrain(neural_network, perception_processor)