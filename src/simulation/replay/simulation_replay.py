from __future__ import annotations

from src.utils	import Loadable

from typing	import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from src.agent.brain	import Brain


class SimulationReplay(Loadable):
	def __init__(
		self, duration: int, brain: Brain, agents: list[dict[str, Any]], food: list[dict[str, Any]]
	):
		self.duration	: int					= duration
		self.brain		: Brain					= brain
		self.agents		: list[dict[str, Any]]	= agents
		self.food		: list[dict[str, Any]]	= food