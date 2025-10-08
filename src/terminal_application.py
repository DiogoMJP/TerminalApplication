from __future__ import annotations

from src.simulation			import create_simulation
from src.training			import create_training
from src.training.replay	import load_training_replay_from_data

import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import textwrap
from itertools	import product
from math		import sqrt
from pathlib	import Path
from time		import time
from typing		import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.simulation			import Simulation
	from src.training			import Training
	from src.training.replay	import GraphData


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
		with open("saved_parameters/default_params.json", "r") as fp:
			data = json.load(fp)
		self.default_params	: dict[str, Any]			= data["default-params"]
		self.params			: dict[str, Any]			= dict(self.default_params)
		self.trainings		: dict[str, dict[str, Any]]	= data["trainings"]
		self.configs		: dict[str, dict[str, Any]]	= data["configs"]

	def main(self) -> None:
		for training_type, train_params in self.trainings.items():
			for config_name, conf_params in self.configs.items():

				self.params = dict(self.default_params)
				for key, val in train_params.items():
					self.params[key] = val
				for key, val in conf_params.items():
					self.params[key] = val

				if "n-repeats" not in self.params.keys():
					raise Exception(f"{self.__class__.__name__}: Invalid parameter: n-repeats")
				for eating_number in self.params["eating-numbers"]:
					for id in range(self.params["n-repeats"]):
						self.train(training_type, config_name, eating_number, id)
					self.create_graphs_from_training_data(training_type, config_name, eating_number)
					self.generate_distance_sound_graphs(training_type, config_name, eating_number)
					self.generate_distance_change_sound_graphs(training_type, config_name, eating_number)
			self.generate_average_performance_graph(training_type)

		print()
		print("Training completed.")
	
	def train(self, training_type: str, config_file: str, eating_number: int, id: int) -> None:
		print()
		self.params["eating-number"] = eating_number
		print(f"Running training {training_type} with config {config_file}, eating number {eating_number}, and id {id} and configs {self.params}")
		start = time()
		training = create_training("neat-training", self.params)
		training.start_training()
		end = time()
		print(f"Training took {end - start:.2f} seconds")
		average_performance, max_performance = self.get_training_result_performance(training)
		Path(f"saved_data/{training_type}/{config_file}/{eating_number}").mkdir(parents=True, exist_ok=True)
		Path(f"saved_data/{training_type}/{config_file}/{eating_number}/{id}.json").unlink(missing_ok=True)
		with open(f"saved_data/{training_type}/{config_file}/{eating_number}/{id}.json", "w+") as fp:
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
	
	def generate_distance_sound_graphs(
		self, training_type: str, config_file: str, eating_number: int
	) -> None:
		print()
		print(f"Generating distance sound graphs for training {training_type} with config {config_file} and eating number {eating_number}")

		def get_closest_food(x: int, y: int, t: int, sim: Simulation) -> tuple[float, float]:
			reg_min_dist_sq = float('inf'); poi_min_dist_sq = float('inf')
			i = 0
			if len(sim.finished_food) == 0: return reg_min_dist_sq, poi_min_dist_sq
			food = sim.finished_food[i]
			while food.last_time_step < t:
				dx = x - food.x
				dy = y - food.y
				dist_sq = dx * dx + dy * dy
				if food.poisonous:
					if dist_sq < poi_min_dist_sq:
						poi_min_dist_sq = dist_sq
				else:
					if dist_sq < reg_min_dist_sq:
						reg_min_dist_sq = dist_sq
				i += 1
				if i >= len(sim.finished_food): break
				food = sim.finished_food[i]
			return sqrt(reg_min_dist_sq), sqrt(poi_min_dist_sq)

		training_replays = []
		for path in Path(f"saved_data/{training_type}/{config_file}/{eating_number}").iterdir():
			if path.is_file() and path.suffix == ".json":
				with open(path, "r") as fp:
					training_replay = load_training_replay_from_data(json.load(fp))
					if "sound-perception-node" not in training_replay.perception_nodes:
						continue
					if training_replay.n_freq == None or training_replay.n_freq == 0:
						continue
					training_replays += [training_replay]
			else: continue
		if training_replays == []:
			return
		
		training_replay = max(
			training_replays, key=lambda tr: tr.average_performance
		)

		regular_sounds : dict[tuple[int, ...], list[float]] = {
			key: [] for key in product([0, 1], repeat=training_replay.n_freq)
			if not all(v == 0 for v in key)
		}
		poisonous_sounds : dict[tuple[int, ...], list[float]] = {
			key: [] for key in product([0, 1], repeat=training_replay.n_freq)
			if not all(v == 0 for v in key)
		}

		brain = training_replay.brain
		if brain == None:
			raise Exception(f"{self.__class__.__name__}: generate_sound_graphs: Training has not been completed")
		simulations : list[Simulation] = []
		for _ in range(20):
			sim = create_simulation(
				training_replay.simulation_type, training_replay.generate_simulation_parameters(brain)
			)
			sim.start_loop()
			simulations += [sim]
		while not all([sim.finished for sim in simulations]):
			pass
		for sim in simulations:
			for t, gen_sounds in enumerate(sim.sound_history):
				for sound in gen_sounds:
					reg_dist, poi_dist = get_closest_food(sound[1][0], sound[1][1], t, sim)
					if reg_dist != float('inf'):
						regular_sounds[sound[1]] += [reg_dist]
					if poi_dist != float('inf'):
						poisonous_sounds[sound[1]] += [poi_dist]
		for sim in simulations: del sim
	
		labels = [str(key) for key in regular_sounds.keys()]
		fig, ax = plt.subplots(layout="constrained")
		dists = [dists for dists in regular_sounds.values()]
		ax.hist(dists, bins=range(0, 500, 20), color=['#D81B60','#1E88E5','#FFC107'], edgecolor = "white", histtype='bar', label=labels, stacked=True)
		plt.xlabel("Distance to closest food")
		plt.ylabel("Number of sounds")
		plt.title("Sound distances to closest regular food")
		plt.legend(title="Sound Channels")
		path = f"saved_data/{training_type}/{config_file}/{eating_number}/regular_sound_.svg"
		Path(path).unlink(missing_ok=True)
		plt.savefig(path, format="svg")
		if training_replay.poisonous_food_rate != None and training_replay.poisonous_food_rate > 0:
			fig, ax = plt.subplots(layout="constrained")
			dists = [dists for dists in poisonous_sounds.values()]
			ax.hist(dists, bins=range(0, 500, 20), color=['#D81B60','#1E88E5','#FFC107'], edgecolor = "white", histtype='bar', label=labels, stacked=True)
			plt.xlabel("Distance to closest food")
			plt.ylabel("Number of sounds")
			plt.title("Sound distances to closest poisonous food")
			plt.legend(title="Sound Channels")
			path = f"saved_data/{training_type}/{config_file}/{eating_number}/poisonous_sound_.svg"
			Path(path).unlink(missing_ok=True)
			plt.savefig(path, format="svg")
		plt.close('all')
	
	def generate_distance_change_sound_graphs(
		self, training_type: str, config_file: str, eating_number: int
	) -> None:
		print()
		print(f"Generating distance change sound graphs for training {training_type} with config {config_file} and eating number {eating_number}")

		def distance_average(distances: list[float]) -> float:
			if len(distances) == 0: return 0.0
			return sum([d/(d+1) for d in distances])/sum([1/(d+1) for d in distances])

		training_replays = []
		for path in Path(f"saved_data/{training_type}/{config_file}/{eating_number}").iterdir():
			if path.is_file() and path.suffix == ".json":
				with open(path, "r") as fp:
					training_replay = load_training_replay_from_data(json.load(fp))
					if "sound-perception-node" not in training_replay.perception_nodes:
						continue
					if training_replay.n_freq == None or training_replay.n_freq == 0:
						continue
					training_replays += [training_replay]
			else: continue
		if training_replays == []:
			return
		
		training_replay = max(
			training_replays, key=lambda tr: tr.average_performance
		)

		sound_distances : dict[tuple[int, ...], list[list[float]]] = {
			key: [[] for _ in range(20)] for key in product([0, 1], repeat=training_replay.n_freq)
			if not all(v == 0 for v in key)
		}

		brain = training_replay.brain
		if brain == None:
			raise Exception(f"{self.__class__.__name__}: generate_sound_graphs: Training has not been completed")
		simulations : list[Simulation] = []
		for _ in range(20):
			sim = create_simulation(
				training_replay.simulation_type, training_replay.generate_simulation_parameters(brain)
			)
			sim.start_loop()
			simulations += [sim]
		while not all([sim.finished for sim in simulations]):
			pass

		for sim in simulations:
			for t, gen_sounds in enumerate(sim.sound_history):
				for sound in gen_sounds:
					for agent in sim.agents:
						if agent.id == sound[0]:
							continue
						for tick in range(t, min(t+20, agent.last_time_step)):
							try:
								sound_distances[sound[2]][tick - t] += [sqrt(
									(agent.get_from_state("x") - sound[1][0])**2 +
									(agent.get_from_state("y") - sound[1][1])**2
								)]
							except: 
								print(sound); input()
		for sim in simulations: del sim

		average_sound_distances : dict[tuple[int, ...], list[float]] = {
			sound: [distance_average(dists) for dists in dist_lists]
			for sound, dist_lists in sound_distances.items()
		}
		average_sound_distances : dict[tuple[int, ...], list[float]] = {
			sound: [d/max(dists) for d in dists] if max(dists) > 0 else dists
			for sound, dists in average_sound_distances.items()
		}

		colors = ['#D81B60','#1E88E5','#FFC107']
		for i, (sound, dists) in enumerate(average_sound_distances.items()):
			fig, ax = plt.subplots(layout="constrained")
			ax.plot([i for i in range(len(dists))], dists, colors[i])
			plt.xlabel("Time since sound (ticks)")
			plt.ylabel("Average distance to sound origin")
			plt.title(f"Sound {sound} average distance to sound origin over time")
			path = f"saved_data/{training_type}/{config_file}/{eating_number}/sound_{sound}_distance_change.svg"
			Path(path).unlink(missing_ok=True)
			plt.savefig(path, format="svg")
		plt.close('all')

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
	
	def generate_average_performance_graph(self, training_type: str) -> None:
		with open(f"saved_data/{training_type}/performance.json", "r") as fp:
			vals: dict[str, dict[str, dict[str, list[float]]]] = json.load(fp)
		vals = {" ".join(key.split(" ")[1:]): val for key, val in vals.items()}
		agents = tuple(vals[list(vals.keys())[0]].keys())
		print()
		print(f"Training {training_type} average performance: {vals}")
		fig, ax = plt.subplots(layout="constrained")
		colors = ['#D81B60','#1E88E5','#FFC107']
		for j, i in enumerate(agents):
			x_vals = list(vals.keys())
			y_vals = [
				sum(vals[key][i]["average-performance"])/len(vals[key][i]["average-performance"])
				for key in vals.keys()
			]
			stdev = [
				sqrt(
					sum([(val - y_vals[j])**2 for val in vals[key][i]["average-performance"]]) /
					(len(vals[key][i]["average-performance"]) - 1)
				)
				for j, key in enumerate(vals.keys())
			]
			ax.plot(x_vals, y_vals, colors[j], label=f"{i}")
			ax.fill_between(
				x_vals, [val - std for val, std in zip(y_vals, stdev)],
				[val + std for val, std in zip(y_vals, stdev)], color=colors[j], alpha=.15
			)
		wrap_labels(ax, list(vals.keys()), 15)
		ax.legend(title="Eating Agents")
		plt.xlabel("Configurations")
		plt.ylabel("Average Duration (time steps)")
		fig.suptitle("Average agent performance")
		Path(f"saved_data/{training_type}/average_performance.svg").unlink(missing_ok=True)
		fig.savefig(f"saved_data/{training_type}/average_performance.svg", format="svg")
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
		ax.plot([i for i in range(len(data["data"]))], data["data"], '#D81B60')
		plt.xlabel(data["x-label"])
		plt.ylabel(data["y-label"])
		plt.title(data["title"])
		Path(path + f"/{data['filename']}.svg").unlink(missing_ok=True)
		plt.savefig(path + f"/{data['filename']}.svg", format="svg")
		plt.close('all')

	def create_graphs_from_training_data(self, training_type: str, config_file: str, eating_number: int) -> None:
		print()
		print(f"Generating graphs for training {training_type} with config {config_file} and eating number {eating_number}")
		data_lists = []
		average_performance = []
		max_performance = []
		for path in Path(f"saved_data/{training_type}/{config_file}/{eating_number}").iterdir():
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
			self.create_graph(graph_data, f"saved_data/{training_type}/{config_file}/{eating_number}")
		sim_type_performance_path = Path(f"saved_data/{training_type}/performance.json")
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