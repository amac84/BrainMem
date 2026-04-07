# BrainMem Architecture Alignment Matrix

This document tracks how the implementation maps to the brain-inspired architecture.

## 1) Encoding Gate
- **Implemented:** `src/brainmem/encoding_gate.py`
- Uses weighted selective encoding:
  - novelty
  - emotional salience
  - goal relevance
  - prediction error
  - unresolved tension
  - repetition
  - explicit emphasis
  - social importance
  - decision irreversibility
- Produces explainable score breakdown for every promotion decision.

## 2) Event Segmentation
- **Implemented:** `src/brainmem/event_segmentation.py`
- Boundary score considers:
  - people, project, tool, place, emotion, mode, action, surprise
- Consolidation uses this to segment daily stream into event-shaped memory.

## 3) Hierarchical Compression
- **Implemented (partial):**
  - daily summaries with gist + anchors in `memory/summaries/daily/`
  - generated via `src/brainmem/consolidation_job.py`
- **Pending depth:** richer weekly schemas and long-arc synthesis.

## 4) Cue-Based Recall
- **Implemented:** cue + state weighted scoring in `src/brainmem/recall_scoring.py`
- Cue bundles include people/project/action/emotion/place/tool/token cues.

## 5) Associative Recall + Interference Control
- **Implemented:**
  - sparse cue-memory graph: `src/brainmem/associative_graph.py`
  - pattern-separation hints for near duplicates
  - competition-set inhibition: `src/brainmem/competition_inhibition.py`

## 6) Reconsolidation
- **Implemented:** `src/brainmem/reconsolidation.py`
  - recalls labilize memories
  - patch queue in `memory/reconsolidation/queue/`
  - commit/defer/conflict decisions
  - claim graph updated in `indexes/claim_graph.json`

## 7) Open Loops
- **Implemented:** `src/brainmem/open_loops.py`
  - open-loop objects persisted in `memory/open_loops/`
  - cue-based trigger matching
  - tension-weighted activation score

## 8) Action Scripts
- **Implemented:** `src/brainmem/action_scripts.py`
  - script extraction/upsert from events
  - script ranking for execution-like recall contexts

## 9) Forgetting and Inhibition
- **Implemented:** `src/brainmem/forgetting.py`
  - decay
  - renormalization
  - reversible archive move for weak memories
- Integrated into daily consolidation.

## 10) Identity / Goal Prior
- **Partial:** project and goal cues strongly influence encoding and recall.
- **Pending depth:** explicit identity belief graph and conflict files.

## 11) Predictive / Simulative Memory Use
- **Implemented:** `src/brainmem/simulation_planner.py`
  - builds state-action transition graph from event memories
  - rollout proposals with evidence anchors
  - exposed via CLI `simulate`.

## Explainability and Auditability
- Canonical artifacts remain Markdown.
- Derived indexes are JSON and rebuildable.
- Retrieval includes score breakdown, graph activation, open-loop/script boosts, and separation hints.

