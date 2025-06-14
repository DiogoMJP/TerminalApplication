from src.simulation		import create_simulation
from src.training		import create_training, Training

import json
import matplotlib.pyplot as plt
import textwrap
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
	"perception-processor-type"	: None,
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

def wrap_labels(ax, labels, width, break_long_words=False):
	ax.set_xticks(labels)
	for i, label in enumerate(labels):
		labels[i] = textwrap.fill(label, width=width, break_long_words=break_long_words)
	ax.set_xticklabels(labels, rotation=0)


class TerminalApplication(object):
	def __init__(self):
		# remove_directory_tree(Path("saved_data"))
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
					self.create_graph_from_training_data(training_type, config_file, eating_number)
			self.generate_average_performance_graph(training_type)
		print()
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
		average_performance, max_performance = self.get_training_result_performance(training)
		Path(f"saved_data/{training_type}/{config_file}/{eating_number}.json").unlink(missing_ok=True)
		Path(f"saved_data/{training_type}/{config_file}").mkdir(parents=True, exist_ok=True)
		with open(f"saved_data/{training_type}/{config_file}/{eating_number}.json", "w+") as fp:
			json.dump(training.to_dict() | {
				"duration" : end - start,
				"average-performance" : average_performance,
				"max-performance" : max_performance
			}, fp, separators=(',', ':'))
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
	
	def generate_average_performance_graph(self, training_type: str) -> None:
		start_directory = Path(f"saved_data/{training_type}")
		vals = {}
		agents = ()
		for config in start_directory.iterdir():
			if config.is_dir():
				config_name = " ".join(config.name.split("_")[1:])
				if config_name not in vals: vals[config_name] = {}
				for file in config.iterdir():
					if file.is_file():
						agent_name = file.name.split(".")[0]
						if agent_name not in agents: agents += (agent_name,)
						with open(file, "r") as fp:
							vals[config_name][agent_name] = json.load(fp)["average-performance"]
		print()
		print(f"Training {training_type} average performance: {vals}")
		fig, ax = plt.subplots(layout="constrained")
		for i in agents:
			ax.plot(vals.keys(), [vals[key][i] for key in vals.keys()], label=f"{i}")
		wrap_labels(ax, list(vals.keys()), 20)
		ax.legend()
		fig.suptitle("Average agent performance")
		fig.savefig(f"saved_data/{training_type}/average_performance.png")
		plt.close('all')
	
	def create_fitness_plot(self, training_type: str, config_file: str, eating_number: int) -> None:
		file_name = f"saved_data/{training_type}/{config_file}/{eating_number}.json"
		with open(file_name, "r") as fp:
			data = json.load(fp)
		fitness_data = [
			sum([
				sum([1 for food in sim["food"] if food["eaten"]])
				for sim in gen.values()
			]) / len(gen.values())
			for _, gen in data["simulations"].items()
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(fitness_data))], fitness_data)
		plt.xlabel("Generation")
		plt.ylabel("Average Fitness")
		plt.title("Average Fitness Over Generations")
		plt.savefig(file_name.replace(".json", "_fitness.png"))

	def create_node_plot(self, training_type: str, config_file: str, eating_number: int) -> None:
		file_name = f"saved_data/{training_type}/{config_file}/{eating_number}.json"
		with open(file_name, "r") as fp:
			data = json.load(fp)
		node_data = [
			sum([
				len(sim["brain"]["network"]["node-evals"]) +
				len(sim["brain"]["network"]["inputs"])
				for sim in gen.values()
			]) / len(gen.values())
			for _, gen in data["simulations"].items()
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(node_data))], node_data)
		plt.xlabel("Generation")
		plt.ylabel("Average Number of Nodes")
		plt.title("Average Number of Nodes Over Generations")
		plt.savefig(file_name.replace(".json", "_nodes.png"))

	def create_duration_plot(self, training_type: str, config_file: str, eating_number: int) -> None:
		# simulation_data = []
		# for _, gen in data["simulations"].items():
		# 	sim = [
		# 		sim["duration"]
		# 		for sim in gen.values()
		# 	]
		# 	sim.sort(reverse=True)
		# 	simulation_data += [sim[:10]]
		# duration_data = [sum(sim) / len(sim) for sim in simulation_data]
		file_name = f"saved_data/{training_type}/{config_file}/{eating_number}.json"
		with open(file_name, "r") as fp:
			data = json.load(fp)
		duration_data = [
			sum([
				sim["duration"] for sim in gen.values()
			]) / len(gen.values())
			for _, gen in data["simulations"].items()
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(duration_data))], duration_data)
		plt.xlabel("Generation")
		plt.ylabel("Average Duration (time steps)")
		plt.title("Average Duration of Simulations Over Generations")
		plt.savefig(file_name.replace(".json", "_duration.png"))

	def create_graph_from_training_data(self, training_type: str, config_file: str, eating_number: int) -> None:
		print()
		print(f"Generating graphs for training {training_type} with config {config_file} and eating number {eating_number}")
		self.create_fitness_plot(training_type, config_file, eating_number)
		self.create_node_plot(training_type, config_file, eating_number)
		self.create_duration_plot(training_type, config_file, eating_number)
		plt.close('all')