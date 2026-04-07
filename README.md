# BrainMem

BrainMem is a Markdown-first, brain-inspired personal assistant memory engine.

It implements:
- selective encoding gate (salience/novelty/goal/tension weighted),
- event segmentation and daily summaries with anchors,
- cue/state recall with associative graph activation,
- competition/inhibition controls,
- reconsolidation queue + claim graph,
- open-loop trigger boosting,
- action-script memory,
- forgetting/archival pass,
- experience-graph simulation for planning.

## Quickstart

```bash
python3 -m pip install pytest
python3 -m pytest tests/unit/test_encoding_gate.py -q
```

## CLI examples

Ingest an event:

```bash
PYTHONPATH=src python3 -m brainmem.cli --root /workspace ingest \
  --text "Need to call Maya to close Atlas vendor handoff loop tomorrow" \
  --people "Maya" \
  --project "Atlas" \
  --tool "laptop" \
  --mode "planning" \
  --emotion "worried" \
  --action "call" \
  --novelty 0.8 --salience 0.8 --goal-relevance 0.9 --surprise 0.6 \
  --unresolved-tension 0.9 --emphasis 0.9
```

Recall by cues + state:

```bash
PYTHONPATH=src python3 -m brainmem.cli --root /workspace recall \
  --cues "vendor,handoff,call" \
  --people "Maya" \
  --project "Atlas" \
  --tool "laptop" \
  --mode "planning" \
  --emotion "worried" \
  --top-k 5
```

Consolidate + reconsolidate + forgetting:

```bash
PYTHONPATH=src python3 -m brainmem.cli --root /workspace consolidate \
  --date 2026-04-07 \
  --boundary-threshold 0.45 \
  --apply-reconsolidation
```

Simulate next actions from experience transitions:

```bash
PYTHONPATH=src python3 -m brainmem.cli --root /workspace simulate \
  --project atlas --mode planning --emotion worried \
  --goal-hint atlas --depth 2 --top-k 3
```

## Test suite

Run all targeted tests:

```bash
python3 -m pytest \
  tests/unit/test_encoding_gate.py \
  tests/unit/test_cue_recall_scoring.py \
  tests/unit/test_event_segmentation.py \
  tests/unit/test_graph_activation.py \
  tests/unit/test_reconsolidation_rules.py \
  tests/unit/test_open_loop_priority.py \
  tests/unit/test_decay_and_inhibition_updates.py \
  tests/integration/test_ingest_recall_cli.py \
  tests/integration/test_daily_summary_anchors.py \
  tests/integration/test_conflict_resolution_commit.py \
  tests/integration/test_reversible_pruning.py \
  tests/simulations/test_similar_memory_interference.py \
  tests/simulations/test_state_dependent_recall.py \
  tests/simulations/test_action_script_retrieval.py \
  tests/simulations/test_planning_rollout_calibration.py \
  tests/simulations/test_adversarial_full_stack.py -q
```
