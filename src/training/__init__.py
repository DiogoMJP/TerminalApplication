from src.training.training		import Training
from src.training.neat_training	import NeatTraining

from typing import Any

def create_training(training_type: str, params: dict[str, Any]) -> Training:
	try:	
		if training_type == "neat-training":
			return NeatTraining.create_from_parameters(params)
		else:
			raise Exception(f"Invalid training type: {training_type}")
	except Exception as e: raise

def get_training_parameters(training_type: str) -> tuple[tuple[str, type], ...]:
	if training_type == "neat-training":
		return NeatTraining.get_parameters()
	else:
		raise Exception(f"Invalid training type: {training_type}")

def get_training_types() -> list[str]:
	return ["neat-training"]