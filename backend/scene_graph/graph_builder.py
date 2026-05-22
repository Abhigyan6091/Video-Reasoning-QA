"""
Graph builder – constructs temporal, causal, and scene graphs using NetworkX.
Tracks object persistence, action chains, and emotional transitions.
"""
from __future__ import annotations

import json
from typing import Any

import networkx as nx


def build_scene_graph(scenes: list[dict], analyses: list[dict]) -> dict:
    """
    Build three interconnected graph representations:
      - temporal_graph : scenes connected by temporal ordering
      - causal_graph   : inferred causal edges between events
      - entity_graph   : object/entity persistence across scenes

    Returns a serializable dict with graph data, entity tracking, etc.
    """
    temporal_graph = nx.DiGraph()
    causal_graph = nx.DiGraph()
    entity_graph = nx.Graph()

    # ── 1. Build temporal graph ────────────────────────────────────
    for i, (scene, analysis) in enumerate(zip(scenes, analyses)):
        node_id = scene["id"]
        temporal_graph.add_node(node_id, **{
            "scene_idx": scene["scene_idx"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "summary": analysis.get("summary", ""),
            "objects": analysis.get("objects", []),
            "actions": analysis.get("actions", []),
            "emotions": analysis.get("emotions", []),
            "mood": analysis.get("mood", ""),
            "representative_frame": analysis.get("representative_frame", ""),
            "keyframes": analysis.get("keyframes", []),
        })

        if i > 0:
            prev_id = scenes[i - 1]["id"]
            temporal_graph.add_edge(prev_id, node_id, relation="follows")

    # ── 2. Track entities across scenes ────────────────────────────
    object_appearances: dict[str, list[str]] = {}  # object -> [scene_ids]
    for scene, analysis in zip(scenes, analyses):
        for obj in analysis.get("objects", []):
            obj_norm = obj.lower().strip()
            if obj_norm not in object_appearances:
                object_appearances[obj_norm] = []
            object_appearances[obj_norm].append(scene["id"])

    # Recurring entities (appear in 2+ scenes)
    recurring_entities = {
        obj: sids for obj, sids in object_appearances.items()
        if len(sids) >= 2
    }

    # Build entity graph: connect scenes sharing entities
    for obj, scene_ids in recurring_entities.items():
        entity_graph.add_node(obj, type="entity", scenes=scene_ids)
        for sid in scene_ids:
            entity_graph.add_edge(obj, sid, relation="appears_in")
        # Connect scene pairs via shared entity
        for i in range(len(scene_ids)):
            for j in range(i + 1, len(scene_ids)):
                if not entity_graph.has_edge(scene_ids[i], scene_ids[j]):
                    entity_graph.add_edge(scene_ids[i], scene_ids[j],
                                          shared_entities=[obj])
                else:
                    existing = entity_graph[scene_ids[i]][scene_ids[j]].get("shared_entities", [])
                    existing.append(obj)

    # ── 3. Infer causal edges ──────────────────────────────────────
    action_chains: list[dict] = []
    for i in range(len(analyses) - 1):
        curr = analyses[i]
        nxt  = analyses[i + 1]
        curr_scene = scenes[i]
        nxt_scene  = scenes[i + 1]

        # Look for potential causal links:
        # actions in current scene that could cause events in next scene
        shared_objects = set(o.lower() for o in curr.get("objects", [])) & \
                         set(o.lower() for o in nxt.get("objects", []))

        if shared_objects or curr.get("anomalies") or nxt.get("anomalies"):
            causal_graph.add_edge(
                curr_scene["id"], nxt_scene["id"],
                relation="may_cause",
                shared_objects=list(shared_objects),
                evidence=f"Shared objects: {shared_objects}" if shared_objects else "Anomaly transition",
            )
            action_chains.append({
                "from_scene": curr_scene["scene_idx"],
                "to_scene": nxt_scene["scene_idx"],
                "shared_objects": list(shared_objects),
            })

    # ── 4. Track emotional transitions ─────────────────────────────
    emotional_arc: list[dict] = []
    for scene, analysis in zip(scenes, analyses):
        emotional_arc.append({
            "scene_idx": scene["scene_idx"],
            "scene_id": scene["id"],
            "mood": analysis.get("mood", "neutral"),
            "emotions": analysis.get("emotions", []),
        })

    # ── 5. Serialize graphs ────────────────────────────────────────
    return {
        "temporal_graph": nx.node_link_data(temporal_graph),
        "causal_graph": nx.node_link_data(causal_graph),
        "entity_graph": nx.node_link_data(entity_graph),
        "recurring_entities": recurring_entities,
        "action_chains": action_chains,
        "emotional_arc": emotional_arc,
        "object_appearances": object_appearances,
    }


def get_graph_context_for_questions(graph_data: dict) -> str:
    """
    Produce a structured text summary of the graph for the LLM to use
    when generating questions.
    """
    lines: list[str] = []

    # Temporal flow
    lines.append("=== TEMPORAL FLOW ===")
    temporal = graph_data.get("temporal_graph", {})
    for node in temporal.get("nodes", []):
        lines.append(
            f"Moment {node.get('scene_idx', '?')} "
            f"({node.get('start_time', 0):.1f}s-{node.get('end_time', 0):.1f}s): "
            f"{node.get('summary', 'N/A')}"
        )
        if node.get("representative_frame"):
            lines.append(f"  Visual reference URL: {node['representative_frame']}")
        if node.get("objects"):
            lines.append(f"  Objects: {', '.join(node['objects'])}")
        if node.get("actions"):
            lines.append(f"  Actions: {', '.join(node['actions'])}")
        if node.get("emotions"):
            lines.append(f"  Emotions: {', '.join(node['emotions'])}")

    # Recurring entities
    recurring = graph_data.get("recurring_entities", {})
    if recurring:
        lines.append("\n=== RECURRING ENTITIES ===")
        for entity, scene_ids in recurring.items():
            lines.append(f"  '{entity}' appears in {len(scene_ids)} scenes")

    # Causal links
    chains = graph_data.get("action_chains", [])
    if chains:
        lines.append("\n=== CAUSAL LINKS ===")
        for chain in chains:
            lines.append(
                f"  Scene {chain['from_scene']} → Scene {chain['to_scene']} "
                f"(shared: {', '.join(chain['shared_objects']) if chain['shared_objects'] else 'anomaly'})"
            )

    # Emotional arc
    arc = graph_data.get("emotional_arc", [])
    if arc:
        lines.append("\n=== EMOTIONAL ARC ===")
        for entry in arc:
            lines.append(f"  Scene {entry['scene_idx']}: mood={entry['mood']}")

    return "\n".join(lines)
