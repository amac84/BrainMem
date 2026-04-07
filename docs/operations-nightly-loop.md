# Nightly Consolidation & Maintenance Loop

This document describes the 02:00 local maintenance run for BrainMem.

## Command

```bash
PYTHONPATH=src python3 -m brainmem.cli --root /workspace consolidate \
  --date 2026-04-07 \
  --boundary-threshold 0.45 \
  --apply-reconsolidation \
  --apply-forgetting
```

## What runs

1. **Event segmentation**
   - Stream records are segmented by boundary shifts (people/project/tool/place/mode/emotion/action/surprise).
2. **Daily summary generation**
   - Produces `memory/summaries/daily/YYYY-MM-DD.md` with:
     - gist
     - anchors
     - segment notes
3. **Reconsolidation commit**
   - Applies queued reconsolidation patches with conflict checks.
   - Updates `indexes/claim_graph.json`.
4. **Forgetting pass**
   - Applies decay and renormalization on `indexes/strength_table.json`.
   - Moves low-strength memories into `memory/archive/` (reversible pruning).

## Output fields

`brainmem consolidate` returns:

- `events_considered`
- `segments`
- `summary_path`
- `anchors`
- `gist_lines`
- `reconsolidation` (if enabled)
- `forgetting` (if enabled)

## Operational notes

- Reconsolidation and forgetting are independently toggleable via CLI flags.
- Archive moves are reversible while files remain in `memory/archive`.
- Derived indexes are rebuildable and not canonical memory substrate.
