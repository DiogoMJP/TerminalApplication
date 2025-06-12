from src.food import Food

from typing import Any


class DefaultFood(Food):
	def __init__(
		self, x: int, y: int, eating_number: int, first_time_step: int,
		lifespan: int, detection_radius: int
	):
		super().__init__(x, y, eating_number, first_time_step, lifespan, detection_radius)

	def simulate(self, time_step: int) -> None:
		if self.alive:
			if time_step - self.first_time_step >= self.lifespan or len(self.eaten_by) >= self.eating_number:
				if len(self.eaten_by) >= self.eating_number:
					self.eaten = True
					for agent in self.eaten_by:
						agent.prolong_lifespan(time_step)
				self.alive = False
				self.last_time_step = time_step
			self.eaten_by = []

	def to_dict(self) -> dict[str, Any]:
		return {
			"type": "default-food",
			"first-time-step" : self.first_time_step,
			"last-time-step" : self.last_time_step,
			"eaten" : self.eaten
		}
	
	@staticmethod
	def get_parameters() -> tuple[str, ...]:
		return ("x", "y", "eating-number", "first-time-step", "food-lifespan", "perception-distance")
	
	@staticmethod
	def create_from_parameters(params: dict[str, Any]) -> 'DefaultFood':
		for key in __class__.get_parameters():
			if key not in params:
				raise Exception(f"{__class__.__name__}: Missing required parameter: {key}")
		return DefaultFood(
			params["x"], params["y"], params["eating-number"], params["first-time-step"],
			params["food-lifespan"], params["perception-distance"]
		)