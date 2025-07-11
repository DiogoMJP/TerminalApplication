from __future__ import annotations

from src.agent.brain		import load_brain_from_data
from src.simulation.replay	import load_simulation_replay_from_data
from src.training.replay	import TrainingReplay

from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.simulation.replay	import SimulationReplay
	from src.training.replay	import GraphData


class NeatTrainingReplay(TrainingReplay):
	def __init__(
		self, n_generations: int, width: int, height: int, n_agents: int, agent_type: str,
		agents_lifespan: int, agents_lifespan_extension: int, food_lifespan: int,
		perception_distance: int, eating_distance: int, eating_number: int, max_time_steps: int,
		perception_processor_type: str, simulation_type: str, config_file: str, config_params: dict[str, Any],
		simulations: dict[str, dict[str, dict[str, Any]]], duration: float, average_performance: float,
		max_performance: int
	):
		super().__init__(
			n_generations, width, height, n_agents, agent_type, agents_lifespan, agents_lifespan_extension,
			food_lifespan, perception_distance, eating_distance, eating_number, max_time_steps, 
			perception_processor_type, simulation_type, duration, average_performance, max_performance
		)
		self.config_file				: str	= config_file
		self.config_params				: dict[str, Any]							= config_params
		self.simulations 				: dict[str, dict[str, SimulationReplay]]	= {
			gen_id: {
				sim_id : load_simulation_replay_from_data(sim_data)
				for sim_id, sim_data in gen.items()
			} for gen_id, gen in simulations.items()
		}
	
	def get_fitness_graph_data(self) -> GraphData:
		return {
			"title": "Average Fitness Over Generations",
			"filename": "fitness",
			"x-label": "Generation",
			"y-label": "Average Fitness",
			"data": [
				sum([
					sum([1 for food in sim.food if food["eaten"]])
					for sim in gen.values()
				]) / len(gen.values())
				for _, gen in self.simulations.items()
			]
		}

	def get_node_graph_data(self) -> GraphData:
		return {
			"title": "Average Number of Nodes Over Generations",
			"filename": "nodes",
			"x-label": "Generation",
			"y-label": "Average Number of Nodes",
			"data": [
				sum([sim.brain.get_n_nodes() for sim in gen.values()]) / len(gen.values())
				for _, gen in self.simulations.items()
			]
		}
	
	def get_duration_graph_data(self) -> GraphData:
		return {
			"title": "Average Duration of Simulations Over Generations",
			"filename": "duration",
			"x-label": "Generation",
			"y-label": "Average Duration (time steps)",
			"data": [
				sum([
					sim.duration for sim in gen.values()
				]) / len(gen.values())
				for _, gen in self.simulations.items()
			]
		}

	def get_food_graph_data(self) -> GraphData:
		return {
			"title": "Average Amount of Food Over Each Time Step",
			"filename": "food",
			"x-label": "Time Step",
			"y-label": "Average Amount of Food",
			"data": [sum([
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
		}
	
	def get_graphs_data(self) -> list[GraphData]:
		return [
			self.get_fitness_graph_data(),
			self.get_node_graph_data(),
			self.get_duration_graph_data(),
			self.get_food_graph_data()
		]

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'NeatTrainingReplay':
		training_replay = NeatTrainingReplay(
			data["n-generations"], data["width"], data["height"], data["n-agents"], data["agent-type"],
			data["agents-lifespan"], data["agents-lifespan-extension"], data["food-lifespan"],
			data["perception-distance"], data["eating-distance"], data["eating-number"], data["max-time-steps"],
			data["perception-processor-type"], data["simulation-type"], data["config-file"], data["config-params"],
			data["simulations"], data["duration"], data["average-performance"], data["max-performance"]
		)
		if "food-spawn-rate" in data: training_replay.food_spawn_rate = data["food-spawn-rate"]
		if "n-food" in data: training_replay.n_food = data["n-food"]
		if "brain" in data: training_replay.brain = load_brain_from_data(data["brain"])
		if "n-sensors" in data: training_replay.n_sensors = data["n-sensors"]
		if "fov" in data: training_replay.fov = data["fov"]
		if "n-freq" in data: training_replay.n_freq = data["n-freq"]

		return training_replay