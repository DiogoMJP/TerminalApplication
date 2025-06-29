from src.agent.brain.neural_network.neural_network		import NeuralNetwork
from src.agent.brain.neural_network.neat_neural_network	import NeatNeuralNetwork

from typing import Any


def create_neural_network(neural_network_type: str, params: dict[str, Any]) -> NeuralNetwork:
	try:
		if neural_network_type == "neat-neural-network":
			return NeatNeuralNetwork.create_from_parameters(params)
		else:
			raise Exception(f"Invalid neural network type: {neural_network_type}")
	except Exception as e: raise

def get_neural_network_parameters(neural_network_type: str) -> tuple[str, ...]:
	if neural_network_type == "neat-neural-network":
		return NeatNeuralNetwork.get_parameters()
	else:
		raise Exception(f"Invalid neural network type: {neural_network_type}")

def get_neural_network_types() -> list[str]:
	return ["neat-neural-network"]