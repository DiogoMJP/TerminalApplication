from src.application    import Application
from src.simulation		import get_simulation_parameters, get_simulation_types

import os
from typing	import Any


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


class TerminalApplication(Application):
	def __init__(self):
		self.params				: dict[str, Any]	= dict(DEFAULT_PARAMS)
		self.simulation_types	: list[str]			= get_simulation_types()
		self.config_files		: list[str]			= [
			file for file in os.listdir( "config_files/")
		]
		self.eating_numbers		: list[int]			= list(EATING_NUMBERS)

	def clear_screen(self):
		os.system('cls' if os.name == 'nt' else 'clear')

	def main_menu(self):
		answered = False
		while not answered:
			self.clear_screen()
			print("Select one of the following options:")
			print("\t1. Run training process.")
			print("\t2. Generate graphs from training data.")
			choice = input("Enter your choice: ")
			if choice == "1":
				self.training_menu()
				answered = True
			elif choice == "2":
				self.graphs_menu()
				answered = True
			else:
				print("Invalid input. Please enter numbers only.")
				input("Press <Enter> to continue...")
	
	def training_menu(self):
		sim_types = self.simulation_type_menu()
		
		params = {sim_type: {} for sim_type in sim_types}

		# Select the configuration for each simulation type
		for sim_type in sim_types:
			configs = self.configs_menu()
			params[sim_type]["config_files"] = configs
			for config in configs:
				params = self.params_menu(sim_type, config)
	
	def simulation_type_menu(self) -> list[str]:
		sim_types = []
		# Select the desired simulation types
		answered = False
		while not answered:
			self.clear_screen()
			print("Select the desired simulation type:")
			for i, simulation_type in enumerate(self.simulation_types):
				print(f"\t{i+1:02d}. {simulation_type}")
			choice = input("Enter your choice (space separated for multiple): ")
			try:
				sim_types = [int(sim_type) - 1 for sim_type in choice.split(" ")]
				for sim_type in sim_types:
					if sim_type < 0 or sim_type >= len(self.simulation_types):
						raise IndexError(f"Invalid simulation type index: {sim_type + 1}")
				answered = True
			except:
				print("Invalid input. Please enter numbers only.")
				input("Press <Enter> to continue...")
		return [self.simulation_types[sim_type] for sim_type in sim_types]
	
	def configs_menu(self) -> list[str]:
		configs = []
		# Select the desired configuration files
		answered = False
		while not answered:
			self.clear_screen()
			print("Select the desired configuration files:")
			for i, (config_file, _) in enumerate(self.config_files):
				print(f"\t{i+1:02d}. {config_file}")
			choice = input("Enter your choice (space separated for multiple): ")
			try:
				configs = [int(config) - 1 for config in choice.split(" ")]
				for config in configs:
					if config < 0 or config >= len(self.config_files):
						raise IndexError(f"Invalid configuration index: {config + 1}")
				answered = True
			except:
				print("Invalid input. Please enter numbers only.")
				input("Press <Enter> to continue...")
		return [self.config_files[config][0] for config in configs]

	def params_menu(self, simulation_type: str, config_file: str) -> dict[str, Any]:
		params = {}
		for param in get_simulation_parameters(simulation_type):
			if param in self.params:
				value = input(f"Enter value for {param} (default: {self.params[param]}): ")
				if value.strip() == "":
					params[param] = self.params[param]
				else:
					try:
						params[param] = type(self.params[param])(value)
					except ValueError:
						print(f"Invalid value for {param}. Using default value: {self.params[param]}")
						params[param] = self.params[param]
			else:
				print(f"Warning: Parameter {param} is not recognized. Skipping.")
		return params
	
	def graphs_menu(self):
		pass