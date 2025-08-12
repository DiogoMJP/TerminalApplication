from src.agent.brain.perception_processors.perception_nodes.perception_node 				import PerceptionNode
from src.agent.brain.perception_processors.perception_nodes.angle_distance_perception_node	import AngleDistancePerceptionNode
from src.agent.brain.perception_processors.perception_nodes.eyes_perception_node 			import EyesPerceptionNode
from src.agent.brain.perception_processors.perception_nodes.sound_perception_node 			import SoundPerceptionNode

from typing	import Any


def create_perception_node(perception_node_type: str, params: dict[str, Any]) -> PerceptionNode:
	try:
		if perception_node_type == "angle-distance-perception-node":
			return AngleDistancePerceptionNode.create_from_parameters(params)
		elif perception_node_type == "eyes-perception-node":
			return EyesPerceptionNode.create_from_parameters(params)
		elif perception_node_type == "sound-perception-node":
			return SoundPerceptionNode.create_from_parameters(params)
		else:
			raise Exception(f"Invalid perception processor type: {perception_node_type}")
	except Exception as e: raise

def get_perception_node_parameters(perception_node_type: str) -> tuple[tuple[str, type], ...]:
	if perception_node_type == "angle-distance-perception-node":
		return AngleDistancePerceptionNode.get_parameters()
	elif perception_node_type == "eyes-perception-node":
		return EyesPerceptionNode.get_parameters()
	elif perception_node_type == "sound-perception-node":
		return SoundPerceptionNode.get_parameters()
	else:
		raise Exception(f"Invalid perception node type: {perception_node_type}")

def load_perception_node_from_data(data: dict[str, Any]) -> PerceptionNode:
	if data["type"] == "angle-distance-perception-node":
		return AngleDistancePerceptionNode.load_from_data(data)
	elif data["type"] == "eyes-perception-node":
		return EyesPerceptionNode.load_from_data(data)
	elif data["type"] == "sound-perception-node":
		return SoundPerceptionNode.load_from_data(data)
	else:
		raise Exception(f"Invalid perception node type: {data['type']}")

def get_perception_node_types() -> list[str]:
	return [
		"angle-distance-perception-node",
		"eyes-perception-node",
		"sound-perception-node"
	]