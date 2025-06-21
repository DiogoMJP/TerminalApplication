from __future__ import annotations

from src.agent.brain		import load_brain_from_data
from src.simulation.replay	import load_simulation_replay_from_data
from src.training.replay	import TrainingReplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.simulation.replay	import SimulationReplay


class NeatTrainingReplay(TrainingReplay):
	def __init__(
		self, n_generations: int, width: int, height: int, n_agents: int, agent_type: str,
		agents_lifespan: int, agents_lifespan_extension: int, food_lifespan: int,
		perception_distance: int, eating_distance: int, eating_number: int, max_time_steps: int,
		perception_processor_type: str, simulation_type: str, config_file: str, config_params: dict[str, Any],
		simulations: dict[str, dict[str, dict[str, Any]]]
	):
		super().__init__(
			n_generations, width, height, n_agents, agent_type, agents_lifespan, agents_lifespan_extension,
			food_lifespan, perception_distance, eating_distance, eating_number, max_time_steps, 
			perception_processor_type, simulation_type
		)
		self.config_file				: str	= config_file
		self.config_params				: dict[str, Any]							= config_params
		self.simulations 				: dict[str, dict[str, SimulationReplay]]	= {
			gen_id: {
				sim_id : load_simulation_replay_from_data(sim_data)
				for sim_id, sim_data in gen.items()
			} for gen_id, gen in simulations.items()
		}

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'NeatTrainingReplay':
		training_replay = NeatTrainingReplay(
			data["n-generations"], data["width"], data["height"], data["n-agents"], data["agent-type"],
			data["agents-lifespan"], data["agents-lifespan-extension"], data["food-lifespan"],
			data["perception-distance"], data["eating-distance"], data["eating-number"], data["max-time-steps"],
			data["perception-processor-type"], data["simulation-type"], data["config-file"], data["config-params"],
			data["simulations"]
		)
		if "food-spawn-rate" in data: training_replay.food_spawn_rate = data["food-spawn-rate"]
		if "n-food" in data: training_replay.n_food = data["n-food"]
		if "brain" in data: training_replay.brain = load_brain_from_data(data["brain"])
		if "n-sensors" in data: training_replay.n_sensors = data["n-sensors"]
		if "fov" in data: training_replay.fov = data["fov"]

		return training_replay

	def create_fitness_plot(self, path: str) -> None:
		fitness_data = [
			sum([
				sum([1 for food in sim.food if food["eaten"]])
				for sim in gen.values()
			]) / len(gen.values())
			for _, gen in self.simulations.items()
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(fitness_data))], fitness_data)
		plt.xlabel("Generation")
		plt.ylabel("Average Fitness")
		plt.title("Average Fitness Over Generations")
		plt.savefig(path + "_fitness.png")

	def create_node_plot(self, path: str) -> None:
		node_data = [
			sum([sim.brain.get_n_nodes() for sim in gen.values()]) / len(gen.values())
			for _, gen in self.simulations.items()
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(node_data))], node_data)
		plt.xlabel("Generation")
		plt.ylabel("Average Number of Nodes")
		plt.title("Average Number of Nodes Over Generations")
		plt.savefig(path + "_nodes.png")

	def create_duration_plot(self, path: str) -> None:
		# simulation_data = []
		# for _, gen in data["simulations"].items():
		# 	sim = [
		# 		sim["duration"]
		# 		for sim in gen.values()
		# 	]
		# 	sim.sort(reverse=True)
		# 	simulation_data += [sim[:10]]
		# duration_data = [sum(sim) / len(sim) for sim in simulation_data]
		duration_data = [
			sum([
				sim.duration for sim in gen.values()
			]) / len(gen.values())
			for _, gen in self.simulations.items()
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(duration_data))], duration_data)
		plt.xlabel("Generation")
		plt.ylabel("Average Duration (time steps)")
		plt.title("Average Duration of Simulations Over Generations")
		plt.savefig(path + "_duration.png")
	
	def create_food_plot(self, path: str) -> None:
		time_step_data = [sum([
			sum([
				len([
					1 for food in sim.food
					if food["first-time-step"] <= i and food["last-time-step"] > i
				])
				for sim in gen.values()
			]) for _, gen in self.simulations.items()]) / sum([
					len([1 for sim in gen.values()
						if sim.duration > i
					])
				for _, gen in self.simulations.items()
			])
			for i in range(max([
					max([sim.duration for sim in gen.values()])
				for _, gen in self.simulations.items()
			]))
		]
		_, ax = plt.subplots()
		ax.plot([i for i in range(len(time_step_data))], time_step_data)
		plt.xlabel("Time Step")
		plt.ylabel("Average Amount of Food")
		plt.title("Average Amount of Food Over Each Time Step")
		plt.savefig(path + "_food.png")
	
	def create_graphs(self, path: str) -> None:
		self.create_fitness_plot(path)
		self.create_node_plot(path)
		self.create_duration_plot(path)
		self.create_food_plot(path)
		plt.close('all')