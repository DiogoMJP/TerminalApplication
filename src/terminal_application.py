from __future__ import annotations

from src.agent.brain		import create_brain
from src.simulation			import create_simulation
from src.training			import create_training
from src.training.replay	import load_training_replay_from_data

import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import textwrap
from pathlib	import Path
from time		import time
from typing		import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.training	import Training


DEFAULT_PARAMS = {
	"n-generations"				: 25,
	"width"						: 1000,
	"height"					: 1000,
	"n-agents"					: 15,
	"agent-type"				: "default-agent",
	"agents-lifespan"			: 350,
	"agents-lifespan-extension"	: 350,
	"food-type"					: "default-food",
	"food-lifespan"				: 350,
	"perception-distance"		: 200,
	"eating-distance"			: 15,
	"eating-number"				: None,
	"max-time-steps"			: 3500,
	"config-file"				: None,
	"perception-processor-type"	: None,
	"simulation-type"			: None,
	"food-spawn-rate"			: 0.02,
	"n-food"					: 6,
	"n-sensors"					: 5,
	"fov"						: 160
}

SIMULATION_TYPES = [
	"random-food-simulation",
	"fixed-food-simulation"
]

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
	}),
	("05_eyes_implementation_config", {
		"perception-processor-type" : "eyes-perception-processor",
		"perception-distance"		: 400
	})
]

EATING_NUMBERS = [1, 2, 3]


def remove_directory_tree(start_directory: Path) -> None:
	if not start_directory.exists(): return
	for path in start_directory.iterdir():
		if path.is_file(): path.unlink()
		else: remove_directory_tree(path)
	start_directory.rmdir()

def wrap_labels(ax, labels, width, break_long_words=False):
	ax.set_xticks(labels)
	for i, label in enumerate(labels):
		labels[i] = textwrap.fill(label, width=width, break_long_words=break_long_words)
	ax.set_xticklabels(labels, rotation=0)


class TerminalApplication(object):
	def __init__(self):
		# remove_directory_tree(Path("saved_data"))
		self.params				: dict[str, Any]					= dict(DEFAULT_PARAMS)
		self.simulation_types	: list[str]							= list(SIMULATION_TYPES)
		self.config_files		: list[tuple[str, dict[str, Any]]]	= [
			(config[0], dict(config[1])) for config in CONFIGS
		]
		self.eating_numbers		: list[int]							= list(EATING_NUMBERS)

	def main(self) -> None:
		for simulation_type in self.simulation_types:
			for config_file, params in self.config_files:
				for key, val in params.items():
					if key not in self.params.keys():
						raise Exception(f"{self.__class__.__name__}: Invalid parameter: {key}")
					else: self.params[key] = val
				for eating_number in self.eating_numbers:
					self.train(simulation_type, config_file, eating_number)
					self.create_graph_from_training_data(simulation_type, config_file, eating_number)
			self.generate_average_performance_graph(simulation_type)
		print()
		print("Training completed.")
		# self.watch_simulation("random-food-simulation", "01_starting_config", 1)
	
	def train(self, simulation_type: str, config_file: str, eating_number: int) -> None:
		print()
		self.params["simulation-type"] = simulation_type
		self.params["config-file"] = f"config_files/{config_file}"
		self.params["eating-number"] = eating_number
		print(f"Running simulation {simulation_type} with config {config_file} and eating number {eating_number} and configs {self.params}")
		start = time()
		training = create_training("neat-training", self.params)
		training.start_training()
		end = time()
		print(f"Training took {end - start:.2f} seconds")
		average_performance, max_performance = self.get_training_result_performance(training)
		Path(f"saved_data/{simulation_type}/{config_file}").mkdir(parents=True, exist_ok=True)
		for suffix in (".json", "_duration.png", "_fitness.png", "_food.png", "_nodes.png"):
			Path(f"saved_data/{simulation_type}/{config_file}/{eating_number}{suffix}").unlink(missing_ok=True)
		with open(f"saved_data/{simulation_type}/{config_file}/{eating_number}.json", "w+") as fp:
			json.dump(training.to_dict() | {
				"duration" : end - start,
				"average-performance" : average_performance,
				"max-performance" : max_performance
			}, fp, separators=(',', ':'))
		print(f"Average Performance: {average_performance}, Maximum Performance: {max_performance}")
		del training

	def get_training_result_performance(self, training: Training) -> tuple[float, float]:
		brain = training.brain
		if brain == None:
			raise Exception(f"{self.__class__.__name__}: get_training_result_performance: Training has not been completed")
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
		for sim in simulations: del sim
		return sum(durations)/len(durations), max(durations)
	
	def generate_average_performance_graph(self, simulation_type: str) -> None:
		start_directory = Path(f"saved_data/{simulation_type}")
		vals = {}
		agents = ()
		for config in start_directory.iterdir():
			if config.is_dir():
				config_name = " ".join(config.name.split("_")[1:])
				if config_name not in vals: vals[config_name] = {}
				for file in config.iterdir():
					if file.is_file() and file.name.endswith(".json"):
						agent_name = file.name.split(".")[0]
						if agent_name not in agents: agents += (agent_name,)
						with open(file, "r") as fp:
							vals[config_name][agent_name] = json.load(fp)["average-performance"]
		print()
		print(f"Simulation {simulation_type} average performance: {vals}")
		fig, ax = plt.subplots(layout="constrained")
		for i in agents:
			ax.plot(vals.keys(), [vals[key][i] for key in vals.keys()], label=f"{i}")
		wrap_labels(ax, list(vals.keys()), 15)
		ax.legend()
		plt.xlabel("Configuration Files")
		plt.ylabel("Average Duration (time steps)")
		fig.suptitle("Average agent performance")
		fig.savefig(f"saved_data/{simulation_type}/average_performance.png")
		plt.close('all')

	def create_graph_from_training_data(self, simulation_type: str, config_file: str, eating_number: int) -> None:
		print()
		print(f"Generating graphs for training {simulation_type} with config {config_file} and eating number {eating_number}")
		with open(f"saved_data/{simulation_type}/{config_file}/{eating_number}.json", "r") as fp:
			training_replay = load_training_replay_from_data(json.load(fp)	)
		training_replay.create_graphs(f"saved_data/{simulation_type}/{config_file}/{eating_number}")