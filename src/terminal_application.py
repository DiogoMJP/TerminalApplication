from __future__ import annotations

from src.simulation			import create_simulation
from src.training			import create_training
from src.training.replay	import load_training_replay_from_data

import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import textwrap
from math		import sqrt
from pathlib	import Path
from time		import time
from typing		import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.training			import Training
	from src.training.replay	import GraphData


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
					for id in range(10):
						self.train(simulation_type, config_file, eating_number, id)
					self.create_graphs_from_training_data(simulation_type, config_file, eating_number)
			self.generate_average_performance_graph(simulation_type)
		print()
		print("Training completed.")
		# self.watch_simulation("random-food-simulation", "01_starting_config", 1)
	
	def train(self, simulation_type: str, config_file: str, eating_number: int, id: int) -> None:
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
		Path(f"saved_data/{simulation_type}/{config_file}/{eating_number}").mkdir(parents=True, exist_ok=True)
		for suffix in (".json", "_duration.png", "_fitness.png", "_food.png", "_nodes.png"):
			Path(f"saved_data/{simulation_type}/{config_file}/{eating_number}{suffix}").unlink(missing_ok=True)
		with open(f"saved_data/{simulation_type}/{config_file}/{eating_number}/{id}.json", "w+") as fp:
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

	def generate_graphs(self, starting_directory: str):
		start_path = Path(f"saved_data/{starting_directory}")
		for config in start_path.iterdir():
			if config.is_dir():
				self.generate_graphs(f"{starting_directory}/{config.name}")
			elif config.suffix == ".json":
				print(f"Generating graphs for {config}")
				with open(config, "r") as fp:
					training_replay = load_training_replay_from_data(json.load(fp))
				training_replay.create_graphs(f"{start_path}/{config.stem}")
	
	def generate_average_performance_graph(self, simulation_type: str) -> None:
		with open(f"saved_data/{simulation_type}/performance.json", "r") as fp:
			vals: dict[str, dict[str, dict[str, list[float]]]]= json.load(fp)
		vals = {" ".join(key.split("_")[1:]): val for key, val in vals.items()}
		agents = tuple(vals[list(vals.keys())[0]].keys())
		print()
		print(f"Simulation {simulation_type} average performance: {vals}")
		fig, ax = plt.subplots(layout="constrained")
		for i in agents:
			x_vals = vals.keys()
			y_vals = [
				sum(vals[key][i]["average-performance"])/len(vals[key][i]["average-performance"])
				for key in vals.keys()
			]
			stdev = [
				sqrt(sum([(val - y_vals[j])**2 for val in vals[key][i]["average-performance"]]) / (len(vals[key][i]["average-performance"]) - 1))
				for j, key in enumerate(vals.keys())
			]
			ax.plot(x_vals, y_vals, label=f"{i}")
			ax.fill_between(x_vals, [val - std for val, std in zip(y_vals, stdev)], [val + std for val, std in zip(y_vals, stdev)], color='b', alpha=.15)
		wrap_labels(ax, list(vals.keys()), 15)
		ax.legend()
		plt.xlabel("Configuration Files")
		plt.ylabel("Average Duration (time steps)")
		fig.suptitle("Average agent performance")
		fig.savefig(f"saved_data/{simulation_type}/average_performance.png")
		plt.close('all')

	def join_graph_data(self, data_list: list[GraphData]) -> GraphData:
		data = [
			[l["data"][i] for l in data_list if len(l["data"]) > i]
			for i in range(max([len(l["data"]) for l in data_list]))
		]
		data = [sum(l)/len(l) for l in data]
		return {
			"title": data_list[0]["title"],
			"filename": data_list[0]["filename"],
			"x-label": data_list[0]["x-label"],
			"y-label": data_list[0]["y-label"],
			"data": data
		}
	
	def create_graph(self, data: GraphData, path: str) -> None:
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(data["data"]))], data["data"])
		plt.xlabel(data["x-label"])
		plt.ylabel(data["y-label"])
		plt.title(data["title"])
		plt.savefig(path + f"/{data['filename']}.png")
		plt.close('all')

	def create_graphs_from_training_data(self, simulation_type: str, config_file: str, eating_number: int) -> None:
		print()
		print(f"Generating graphs for training {simulation_type} with config {config_file} and eating number {eating_number}")
		data_lists = []
		average_performance = []
		max_performance = []
		for path in Path(f"saved_data/{simulation_type}/{config_file}/{eating_number}").iterdir():
			if path.is_file() and path.suffix == ".json":
				with open(path, "r") as fp:
					training_replay = load_training_replay_from_data(json.load(fp))
				graphs_data = training_replay.get_graphs_data()
				if data_lists == []:
					data_lists = [[graph_data] for graph_data in graphs_data]
				else:
					for i in range(len(data_lists)):
						data_lists[i] += [graphs_data[i]]
				average_performance += [training_replay.average_performance]
				max_performance += [training_replay.max_performance]
		data_lists = [self.join_graph_data(data_list) for data_list in data_lists]
		for graph_data in data_lists:
			self.create_graph(graph_data, f"saved_data/{simulation_type}/{config_file}/{eating_number}")
		sim_type_performance_path = Path(f"saved_data/{simulation_type}/performance.json")
		if sim_type_performance_path.exists():
			with open(sim_type_performance_path, "r") as fp:
				sim_type_performance = json.load(fp)
		else:
			sim_type_performance = {}
		if config_file not in sim_type_performance:
			sim_type_performance[config_file] = {}
		sim_type_performance[config_file][eating_number] = {
			"average-performance": average_performance,
			"max-performance": max_performance
		}
		with open(sim_type_performance_path, "w") as fp:
			json.dump(sim_type_performance, fp, indent=2)