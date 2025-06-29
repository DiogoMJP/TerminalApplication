from src.simulation.simulation				import Simulation
from src.simulation.random_food_simulation	import RandomFoodSimulation
from src.simulation.fixed_food_simulation	import FixedFoodSimulation

from typing	import Any


def create_simulation(simulation_type: str, params: dict[str, Any]) -> Simulation:
	try:
		if simulation_type == "random-food-simulation":
			return RandomFoodSimulation.create_from_parameters(params)
		elif simulation_type == "fixed-food-simulation":
			return FixedFoodSimulation.create_from_parameters(params)
		else:
			raise Exception(f"Invalid simulation type: {simulation_type}")
	except Exception as e: raise

def get_simulation_parameters(simulation_type: str) -> tuple[tuple[str, type], ...]:
	if simulation_type == "random-food-simulation":
		return RandomFoodSimulation.get_parameters()
	elif simulation_type == "fixed-food-simulation":
		return FixedFoodSimulation.get_parameters()
	else:
		raise Exception(f"Invalid simulation type: {simulation_type}")

def get_simulation_types() -> list[str]:
	return [
		"random-food-simulation",
		"fixed-food-simulation"
	]