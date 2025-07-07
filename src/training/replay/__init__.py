from src.training.replay.training_replay		import TrainingReplay
from src.training.replay.neat_training_replay	import NeatTrainingReplay

from typing				import Any, TypedDict


GraphData = TypedDict("GraphData", {
	"title": str,
	"filename": str,
	"x-label": str,
	"y-label": str,
	"data": list[float]
})

def load_training_replay_from_data(data: dict[str, Any]) -> TrainingReplay:
	if data["type"] == "neat-training":
		return NeatTrainingReplay.load_from_data(data)
	else:
		raise Exception(f"Invalid training type: {data['type']}")