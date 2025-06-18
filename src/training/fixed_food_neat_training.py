from __future__	import annotations

from src.agent.brain	import create_brain
from src.simulation		import create_simulation
from src.training		import Training

import neat
from neat	import Config, DefaultGenome
from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain
	from src.simulation		import Simulation


class FixedFoodNeatTraining(Training):
	def __init__(
		self, n_generations: int, width: int, height: int, n_agents: int, agent_type: str,
		agents_lifespan: int, agents_lifespan_extension: int, food_lifespan: int,
		perception_distance: int, eating_distance: int, eating_number: int, max_time_steps: int,
		config_file: str, perception_processor_type: str, n_food: int, 
	):
		super().__init__(
			n_generations, width, height, n_agents, agent_type, agents_lifespan, agents_lifespan_extension,
			food_lifespan, perception_distance, eating_distance, eating_number, max_time_steps,
			perception_processor_type, "fixed-food-simulation"
		)
		self.config_file				: str	= config_file

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
	
	def start_training(self) -> None:
		# Load configuration.
		config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation,
							self.config_file)

		# Create the population, which is the top-level object for a NEAT run.
		pop = neat.Population(config)
		pop.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		pop.add_reporter(stats)

		# Run for up to 30 generations.
		winner = pop.run(self.eval_genomes, self.n_generations)

		self.brain = create_brain("neat-brain", {
			"perception-processor-type" : self.perception_processor_type,
			"neat-neural-network" : neat.nn.FeedForwardNetwork.create(winner, config)
		} | self.generate_perception_processor_parameter()) 
	
	def eval_genomes(
			self, genomes: list[tuple[int, DefaultGenome]], config: Config
	) -> None:
		self.simulations[str(self.generation)] = {}
		for id, genome in genomes:
			brain = create_brain("neat-brain", {
				"perception-processor-type" : self.perception_processor_type,
				"neat-neural-network" : neat.nn.FeedForwardNetwork.create(genome, config)
			} | self.generate_perception_processor_parameter()) 
			sim = create_simulation("fixed-food-simulation", self.generate_simulation_parameters(brain))
			sim.start_loop()
			self.simulations[str(self.generation)][str(id)] = (sim, genome)
		while not all([sim.finished for sim, _ in self.simulations[str(self.generation)].values()]):
			pass
		for id, pair in self.simulations[str(self.generation)].items():
			pair[1].fitness = pair[0].get_n_eaten_food()

		self.generation += 1
	
	def generate_simulation_parameters(self, brain: Brain) -> dict[str, Any]:
		return {
			"brain" : brain,
			"width" : self.width,
			"height" : self.height,
			"n-agents" : self.n_agents,
			"agent-type" : self.agent_type,
			"agents-lifespan" : self.agents_lifespan,
			"agents-lifespan-extension" : self.agents_lifespan_extension,
			"food-lifespan" : self.food_lifespan,
			"perception-distance" : self.perception_distance,
			"eating-distance" : self.eating_distance,
			"eating-number" : self.eating_number,
			"max-time-steps" : self.max_time_steps,
			"n-food" : self.n_food
		}

	def get_simulation(self, generation: str, simulation: str) -> Simulation|None:
		if generation in self.simulations:
			if simulation in self.simulations[generation]:
				return self.simulations[generation][simulation][0]
			else: return None
		else: return None
	
	def to_dict(self) -> dict[str, Any]:
		return {
			"type" : "random-food-neat-training",
			"config-file" : self.config_file,
			"n-food" : self.n_food,
			"simulations" : {
				gen_id : {
					sim_id : sim[0].to_dict() for sim_id, sim in gen.items()
				} for gen_id, gen in self.simulations.items()
			},
			"config-params" : self.config_params
		} | super().to_dict()
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return (
			"n-generations", "width", "height", "n-agents", "agent-type", "agents-lifespan",
			"agents-lifespan-extension", "food-lifespan", "perception-distance", "eating-distance",
			"eating-number", "max-time-steps", "config-file", "perception-processor-type", "n-food"
		)
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'FixedFoodNeatTraining':
		for key in __class__.get_parameters():
			if key not in params:
				raise Exception(f"Missing required parameter: {key}")
		training = FixedFoodNeatTraining(
			params["n-generations"], params["width"], params["height"], params["n-agents"], params["agent-type"],
			params["agents-lifespan"], params["agents-lifespan-extension"], params["food-lifespan"],
			params["perception-distance"], params["eating-distance"], params["eating-number"], params["max-time-steps"],
			params["config-file"], params["perception-processor-type"], params["n-food"]
		)
		if "n-sensors" in params: training.n_sensors = params["n-sensors"]
		if "fov" in params: training.fov = params["fov"]
		return training