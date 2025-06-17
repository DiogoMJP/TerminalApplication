from src.training.training					import Training
from src.training.random_food_neat_training	import RandomFoodNeatTraining
from src.training.fixed_food_neat_training	import FixedFoodNeatTraining
from src.training.eyes_neat_training		import EyesNeatTraining

from typing import Any, Dict, Tuple

def create_training(training_type: str, params: Dict[str, Any]) -> Training:
	try:	
		if training_type == "random-food-neat-training":
			return RandomFoodNeatTraining.create_from_parameters(params)
		elif training_type == "fixed-food-neat-training":
			return FixedFoodNeatTraining.create_from_parameters(params)
		elif training_type == "eyes-neat-training":
			return EyesNeatTraining.create_from_parameters(params)
		else:
			raise Exception(f"Invalid training type: {training_type}")
	except Exception as e: raise

def get_training_parameters(training_type: str) -> Tuple[str, ...]:
	if training_type == "random-food-neat-training":
		return RandomFoodNeatTraining.get_parameters()
	elif training_type == "fixed-food-neat-training":
			return FixedFoodNeatTraining.get_parameters()
	elif training_type == "eyes-neat-training":
			return EyesNeatTraining.get_parameters()
	else:
		raise Exception(f"Invalid training type: {training_type}")