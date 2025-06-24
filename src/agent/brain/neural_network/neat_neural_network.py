from src.agent.brain.neural_network	import NeuralNetwork

from neat.nn		import FeedForwardNetwork
from scipy.special	import expit as sigmoid
from typing			import Any


class NeatNeuralNetwork(NeuralNetwork):
	def __init__(self, inputs, outputs, node_evals):
		self.inputs = inputs
		self.outputs = outputs
		self.node_evals = node_evals
		self.values = dict((key, 0.0) for key in inputs + outputs)

	def sum(self, lst: list[float]) -> float:
		return sum(lst)

	def sigmoid(self, x: float) -> float:
		return sigmoid(x)

	def activate(self, inputs: tuple[float, ...]) -> list[float]:
		if len(self.inputs) != len(inputs):
			raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.inputs), len(inputs)))

		for k, v in zip(self.inputs, inputs):
			self.values[k] = v

		for node, bias, response, links in self.node_evals:
			node_inputs = []
			for i, w in links:
				node_inputs.append(self.values[i] * w)
			s = self.sum(node_inputs)
			self.values[node] = self.sigmoid(bias + response * s)

		return [self.values[i] for i in self.outputs]
	
	def to_dict(self) -> dict[str, Any]:
		return {
			"type"			: "neat-neural-network",
			"inputs"		: self.inputs,
			"outputs"		: self.outputs,
			"node-evals"	: self.node_evals
		}

	@staticmethod
	def create_from_neat_nn(nn : FeedForwardNetwork) -> 'NeatNeuralNetwork':
		node_evals = [(n, b, r, l) for n, _, _, b, r, l in nn.node_evals]
		return NeatNeuralNetwork(nn.input_nodes, nn.output_nodes, node_evals)
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return ("neat-neural-network",)
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'NeatNeuralNetwork':
		for key in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
		return NeatNeuralNetwork.create_from_neat_nn(params["neat-neural-network"])
	
	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'NeatNeuralNetwork':
		return NeatNeuralNetwork(data["inputs"], data["outputs"], data["node-evals"])