from src.training.training		import Training
from src.training.neat_training	import NeatTraining

from typing import Any, Dict, Tuple

def create_training(training_type: str, params: Dict[str, Any]) -> Training:
	try:	
		if training_type == "neat-training":
			return NeatTraining.create_from_parameters(params)
		else:
			raise Exception(f"Invalid training type: {training_type}")
	except Exception as e: raise

def get_training_parameters(training_type: str) -> Tuple[str, ...]:
	if training_type == "neat-training":
		return NeatTraining.get_parameters()
	else:
		raise Exception(f"Invalid training type: {training_type}")