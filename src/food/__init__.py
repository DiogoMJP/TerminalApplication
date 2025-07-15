from src.food.food			import Food
from src.food.default_food	import DefaultFood

from typing import Any


def create_food(food_type: str, params: dict[str, Any]) -> Food:
	try:
		if food_type == "default-food":
			return DefaultFood.create_from_parameters(params)
		else:
			raise Exception(f"Invalid food type: {food_type}")
	except Exception as e: raise

def get_food_parameters(food_type: str) -> tuple[tuple[str, type], ...]:
	if food_type == "default-food":
		return DefaultFood.get_parameters()
	else:
		raise Exception(f"Invalid food type: {food_type}")

def get_food_types() -> list[str]:
	return ["default-food"]