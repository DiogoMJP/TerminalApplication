from src.agent.brain	import load_brain_from_data
from src.simulation		import create_simulation
from src.training		import create_training, Training

import json
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from typing import Any, Dict, List


DEFAULT_PARAMS = {
	"n-generations"				: 25,
	"width"						: 400,
	"height"					: 400,
	"n-agents"					: 15,
	"agent-type"				: "default-agent",
	"agents-lifespan"			: 350,
	"agents-lifespan-extension"	: 350,
	"food-lifespan"				: 350,
	"perception-distance"		: 100,
	"eating-distance"			: 15,
	"eating-number"				: None,
	"max-time-steps"			: 2000,
	"config-file"				: None,
	"perception-processor-type"	: "food-agent-distance-perception-processor",
	"food-spawn-rate"			: 0.01,
	"n-food"					: 3
}

TRAINING_TYPES = ["random-food-neat-training", "fixed-food-neat-training"]

CONFIGS = [
	("01_starting_config", {
		"perception-processor-type" : "food-agent-distance-perception-processor"
	}),
	("02_weight_range_change_config", {
		"perception-processor-type" : "food-agent-distance-perception-processor"
	}),
	("03_weight_params_change_config", {
		"perception-processor-type" : "food-agent-distance-perception-processor"
	}),
	("04_normalise_input_config", {
		"perception-processor-type" : "normalised-input-perception-processor"
	})
]

EATING_NUMBERS = [1, 2, 3]


def remove_directory_tree(start_directory: Path) -> None:
	if not start_directory.exists(): return
	for path in start_directory.iterdir():
		if path.is_file(): path.unlink()
		else: remove_directory_tree(path)
	start_directory.rmdir()


class TerminalApplication(object):
	def __init__(self):
		remove_directory_tree(Path("saved_data"))
		self.params			: Dict[str, Any]					= dict(DEFAULT_PARAMS)
		self.training_types	: List[str]							= list(TRAINING_TYPES)
		self.config_files	: list[tuple[str, dict[str, Any]]]	= [
			(config[0], dict(config[1])) for config in CONFIGS
		]
		self.eating_numbers	: List[int]							= list(EATING_NUMBERS)

	def main(self) -> None:
		for training_type in self.training_types:
			for config_file, params in self.config_files:
				for key, val in params.items():
					if key not in self.params.keys():
						raise Exception(f"{self.__class__.__name__}: Invalid parameter: {key}")
					else: self.params[key] = val
				for eating_number in self.eating_numbers:
					self.train(training_type, config_file, eating_number)
		print("Training completed.")
	
	def train(self, training_type: str, config_file: str, eating_number: int) -> None:
		print()
		self.params["config-file"] = f"config_files/{config_file}"
		self.params["eating-number"] = eating_number
		print(f"Running training {training_type} with config {config_file} and eating number {eating_number} and configs {self.params}")
		start = time()
		training = create_training(training_type, self.params)
		training.start_training()
		end = time()
		print(f"Training took {end - start:.2f} seconds")
		Path(f"saved_data/{training_type}/{config_file}").mkdir(parents=True, exist_ok=True)
		with open(f"saved_data/{training_type}/{config_file}/{eating_number}_out.json", "w+") as fp:
			json.dump(training.to_dict() | {"duration" : end - start}, fp, separators=(',', ':'))
		average_performance, max_performance = self.get_training_result_performance(training)
		print(f"Average Performance: {average_performance}, Maximum Performance: {max_performance}")

	def get_training_result_performance(self, training: Training) -> None:
		brain = training.brain
		simulations = []
		for _ in range(50):
			sim = create_simulation(
				training.simulation_type, training.generate_simulation_parameters(brain)
			)
			sim.start_loop()
			simulations += [sim]
		while not all([sim.finished for sim in simulations]):
			pass
		durations = [sim.last_time_step for sim in simulations]
		return sum(durations)/len(durations), max(durations)