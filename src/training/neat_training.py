from __future__	import annotations

from src.agent.brain						import create_brain
from src.agent.brain.perception_processors	import create_perception_processor
from src.simulation							import create_simulation
from src.training							import Training

import neat
from os		import remove
from string	import Template
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain.perception_processors	import PerceptionProcessor
	from src.simulation							import Simulation
	from neat									import Config, DefaultGenome


class NeatTraining(Training):
	def __init__(
		self, n_generations: int, width: int, height: int, n_agents: int, agent_type: str,
		agents_lifespan: int, agents_lifespan_extension: int, food_type: str, food_lifespan: int,
		perception_distance: int, eating_distance: int, eating_number: int, max_time_steps: int,
		perception_processor_type: str, simulation_type: str, config_file: str,
		perception_nodes: list[str], perception_processor: PerceptionProcessor
	):
		super().__init__(
			n_generations, width, height, n_agents, agent_type, agents_lifespan, agents_lifespan_extension,
			food_type, food_lifespan, perception_distance, eating_distance, eating_number, max_time_steps, 
			perception_processor_type, simulation_type, perception_nodes, perception_processor
		)
		self.config_file	: str	= config_file

		self.config_params				: dict[str, Any]											= {}
		self.simulations 				: dict[str, dict[str, tuple[Simulation, DefaultGenome]]]	= {}
		self.generation					: int														= 0

		self.process_config()
	
	def process_config(self) -> None:
		with open(self.config_file, "r") as fp:
			for line in fp.readlines():
				line = line.split("#")[0]
				if len(line) > 0:
					line = line.split("=")
					if len(line) == 2:
						try:
							self.config_params[line[0].strip()] = float(line[1].strip())
						except:
							if line[1].strip() in ["True", "False"]:
								self.config_params[line[0].strip()] = line[1].strip() == "True"
							else:
								self.config_params[line[0].strip()] = str(line[1].strip())
		with open("temp/config", "w+") as fp:
			with open(self.config_file, "r") as src_fp:
				src = Template(src_fp.read())
			pattern = {
				"num_inputs"	: self.perception_processor.get_n_input(),
				"num_outputs"	: self.perception_processor.get_n_output()
			}
			fp.write(src.substitute(pattern))


	def start_training(self) -> None:
		# Load configuration.
		config = neat.Config(
			neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
			neat.DefaultStagnation, "temp/config"
		)

		# Create the population, which is the top-level object for a NEAT run.
		pop = neat.Population(config)
		pop.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		pop.add_reporter(stats)

		# Run for up to 30 generations.
		winner = pop.run(self.eval_genomes, self.n_generations)

		self.brain = create_brain("neat-brain", {
			"perception-processor" : self.perception_processor,
			"neat-neural-network" : neat.nn.FeedForwardNetwork.create(winner, config)
		} | self.generate_perception_processor_parameter())
		remove("temp/config")

	def eval_genomes(
			self, genomes: list[tuple[int, DefaultGenome]], config: Config
	) -> None:
		self.simulations[str(self.generation)] = {}
		for id, genome in genomes:
			brain = create_brain("neat-brain", {
				"perception-processor" : self.perception_processor,
				"neat-neural-network" : neat.nn.FeedForwardNetwork.create(genome, config)
			} | self.generate_perception_processor_parameter()) 
			sim = create_simulation(self.simulation_type, self.generate_simulation_parameters(brain))
			self.simulations[str(self.generation)][str(id)] = (sim, genome)
		while not all([sim.finished for sim, _ in self.simulations[str(self.generation)].values()]):
			pass
		for id, pair in self.simulations[str(self.generation)].items():
			pair[1].fitness = pair[0].get_n_eaten_food()

		self.generation += 1

	def get_simulation(self, generation: str, simulation: str) -> Simulation|None:
		if generation in self.simulations:
			if simulation in self.simulations[generation]:
				return self.simulations[generation][simulation][0]
			else: return None
		else: return None
	
	def to_dict(self) -> dict[str, Any]:
		return {
			"type"				: "neat-training",
			"config-file"		: self.config_file,
			"simulations"		: {
				gen_id : {
					sim_id : sim[0].to_dict() for sim_id, sim in gen.items()
				} for gen_id, gen in self.simulations.items()
			},
			"config-params"		: self.config_params
		} | super().to_dict()
	
	@staticmethod
	def get_parameters() -> tuple[tuple[str, type], ...]:
		return (
			("n-generations", int), ("width", int), ("height", int), ("n-agents", int),
			("agent-type", str), ("agents-lifespan", int), ("agents-lifespan-extension", int),
			("food-type", str), ("food-lifespan", int), ("perception-distance", int), ("eating-distance", int),
			("eating-number", int), ("max-time-steps", int), ("perception-processor-type", str),
			("simulation-type", str), ("config-file", str), ("perception-nodes", list)
		)
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'NeatTraining':
		for key, param_type in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
			if not isinstance(params[key], param_type):
				raise Exception(
					f"{__class__.__name__}: Invalid type for parameter '{key}': expected {param_type}, got {type(params[key])}"
				)
		perception_processor = create_perception_processor(
			params["perception-processor-type"], params
		)
		training = NeatTraining(
			params["n-generations"], params["width"], params["height"], params["n-agents"],
			params["agent-type"], params["agents-lifespan"], params["agents-lifespan-extension"],
			params["food-type"], params["food-lifespan"], params["perception-distance"],
			params["eating-distance"], params["eating-number"], params["max-time-steps"],
			params["perception-processor-type"], params["simulation-type"], params["config-file"],
			params["perception-nodes"], perception_processor
		)
		if "food-spawn-rate" in params: training.food_spawn_rate = params["food-spawn-rate"]
		if "n-food" in params: training.n_food = params["n-food"]
		if "poisonous-food-rate" in params: training.poisonous_food_rate = params["poisonous-food-rate"]
		if "poisonous-perception-distance" in params: training.poisonous_perception_distance = params["poisonous-perception-distance"]
		if "normalized" in params: training.normalized = params["normalized"]
		if "n-cones" in params: training.n_cones = params["n-cones"]
		if "fov" in params: training.fov = params["fov"]
		if "see-agents" in params: training.see_agents = params["see-agents"]
		if "see-food" in params: training.see_food = params["see-food"]
		if "see-poisonous-food" in params: training.see_poisonous_food = params["see-poisonous-food"]
		if "see-walls" in params: training.see_walls = params["see-walls"]
		if "n-freq" in params: training.n_freq = params["n-freq"]
		return training