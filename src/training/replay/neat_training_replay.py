from __future__ import annotations

from src.agent.brain		import load_brain_from_data
from src.simulation.replay	import load_simulation_replay_from_data
from src.training.replay	import TrainingReplay

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