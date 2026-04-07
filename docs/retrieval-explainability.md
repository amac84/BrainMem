# Retrieval explainability (Loop 3)

BrainMem retrieval now combines:

1. **Cue/state scoring** (`recall_scoring.py`)
2. **Associative graph activation** (`associative_graph.py`)
3. **Competition/inhibition updates** (`competition_inhibition.py`)

## Explainability fields in recall output

Each result includes `score_breakdown` keys:

- `cue_overlap` — sparse cue Jaccard between query cues and memory cues
- `state_match` — state congruence over place/tool/social/processing mode
- `goal_relevance` — whether project cue aligns with memory project
- `open_loop_activation` — first-class boost for open loops
- `recency_bonus` — bounded recency gain
- `graph_activation` — associative cue-spread activation score
- `pattern_overlap_with_top` — overlap ratio against top activated memory
- `pattern_separation_required` — set to `1.0` for near-duplicate contenders

## Pattern separation behavior

When top activations are highly overlapping (default Jaccard ≥ 0.75), the engine marks contenders for disambiguation and provides a hint with unique cue differences.

This is intended to reduce same-topic/same-person adjacent-day interference.

## Competition + inhibition behavior

After selecting a winner, the engine applies bounded inhibition to competing memories within the leading cue’s competition set:

- decrement strength by `loser_penalty` (default 0.15)
- clamp at `min_strength` (default 0.05)

Strength changes are written to `indexes/strength_table.json` so inhibition is auditable and reversible.
