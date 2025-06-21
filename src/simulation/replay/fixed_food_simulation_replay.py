from __future__ import annotations

from src.agent.brain		import load_brain_from_data
from src.simulation.replay	import SimulationReplay

from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain


class FixedFoodSimulationReplay(SimulationReplay):
	def __init__(
		self, duration: int, brain: Brain, agents: list[dict[str, Any]], food: list[dict[str, Any]]
	):
		super().__init__(duration, brain, agents, food)

	@staticmethod
	def load_from_data(data: dict[str, Any]) -> 'FixedFoodSimulationReplay':
		return FixedFoodSimulationReplay(
			data["duration"], load_brain_from_data(data["brain"]), data["agents"], data["food"]
		)