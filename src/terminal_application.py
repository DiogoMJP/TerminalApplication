from src.simulation	import create_simulation
from src.training	import create_training

import json
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
CONFIG_FILES = [
	"01_starting_config"
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
		self.params			: Dict[str, Any]	= dict(DEFAULT_PARAMS)
		self.training_types	: List[str]			= list(TRAINING_TYPES)
		self.config_files	: List[str]			= list(CONFIG_FILES)
		self.eating_numbers	: List[int]			= list(EATING_NUMBERS)

	def main(self) -> None:
		for training_type in self.training_types:
			for config_file in self.config_files:
				for eating_number in self.eating_numbers:
					self.train(training_type, config_file, eating_number)
		print("Training completed.")
	
	def train(self, training_type: str, config_file: str, eating_number: int) -> None:
		print()
		print(f"Running training {training_type} with config {config_file} and eating number {eating_number}")
		self.params["config-file"] = f"config_files/{config_file}"
		self.params["eating-number"] = eating_number
		start = time()
		training = create_training(training_type, self.params)
		training.start_training()
		end = time()
		print(f"Training took {end - start:.2f} seconds")
		Path(f"saved_data/{training_type}/{config_file}").mkdir(parents=True, exist_ok=True)
		with open(f"saved_data/{training_type}/{config_file}/{eating_number}_out.json", "w+") as fp:
			json.dump(training.to_dict() | {"duration" : end - start}, fp, separators=(',', ':'))