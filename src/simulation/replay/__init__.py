from src.simulation.replay.simulation_replay				import SimulationReplay
from src.simulation.replay.random_food_simulation_replay	import RandomFoodSimulationReplay
from src.simulation.replay.fixed_food_simulation_replay		import FixedFoodSimulationReplay
from src.simulation.replay.poisonous_food_simulation_replay	import PoisonousFoodSimulationReplay

from typing	import Any


def load_simulation_replay_from_data(data: dict[str, Any]) -> SimulationReplay:
	if data["type"] == "random-food-simulation":
		return RandomFoodSimulationReplay.load_from_data(data)
	elif data["type"] == "fixed-food-simulation":
		return FixedFoodSimulationReplay.load_from_data(data)
	elif data["type"] == "poisonous-food-simulation":
		return PoisonousFoodSimulationReplay.load_from_data(data)
	else:
		raise Exception(f"Invalid simulation type: {data['type']}")