# Brain-Inspired Memory Mechanisms and Novel Recall Architectures for a Personal Assistant

## Executive summary

Human memory is best understood as a **control system for future behaviour**, not a passive store: encoding is selective, retrieval is cue- and state-dependent, recall can **modify** what is remembered, and consolidation compresses experience into multi-scale “schemas” that support prediction and planning. citeturn2search8turn3search4turn4search0turn5search4turn6search2

This report translates neuroscience mechanisms—**beyond the basic working/episodic/semantic split**—into implementable architectures for a personal assistant that primarily uses **Markdown files** (with lightweight derived indices) and **does not assume embeddings/vector search are sufficient**. Where retrieval needs “associativity,” the proposed systems use **cue bundles, symbolic features, event boundaries, and graph dynamics** (spreading activation + inhibition), which map more directly onto cognitive models of cue-driven recall and interference than “nearest vector” alone. citeturn1search0turn8search3turn5search23turn8search0

Core proposals (one or more per mechanism) include:

- An **Encoding Gate with Tag-and-Capture credit assignment** (novelty/arousal “primes” weak events into durable memory), inspired by synaptic/behavioural tagging. citeturn4search0turn4search5turn4search13turn5search4turn4search2  
- A **Versioned Reconsolidation Store** where any retrieved memory enters a time-limited “labilized” state and is only committed after consistency checks and nightly consolidation, reflecting reconsolidation and memory transformation. citeturn0search2turn2search23turn2search15turn2search8turn2search0  
- A **Hippocampal Index Graph** (episodes as sparse indices over distributed features) with **pattern completion + pattern separation** and **retrieval-induced forgetting** to reduce interference. citeturn0search8turn8search1turn8search0turn5search23  
- A **State-Dependent Cue Bundle Retriever** implementing encoding specificity, context dependence, and transfer-appropriate processing, so the assistant “remembers different things” in different modes or environments. citeturn1search0turn1search1turn1search7turn1search6  
- A **Replay-and-Rollout simulator** that uses experience-derived transition graphs to support planning (episodic future thinking, route bias to goals) and nightly “replay.” citeturn3search3turn7search10turn3search4turn0search7  
- An **Open-Loop Tension Register** (Zeigarnik/prospective memory) that treats unfinished intentions as first-class memory objects with event/time triggers. citeturn6search0turn6search2turn6search19turn6search7  
- An **Action-Script Memory** grounded in event segmentation and enactment effects, linking recall to actions/tools/outcomes rather than words alone. citeturn7search1turn7search9turn7search4  
- An **Identity/Goal Prior** (self-reference effect; mPFC/DMN) that changes both encoding and retrieval so the system learns “who you are becoming,” not just what happened. citeturn10search16turn10search2turn10search1turn10search9turn9search3  

## Neuroscience mechanisms that matter for recall architectures

A few empirically grounded mechanisms strongly constrain what “human-like memory” implies for a system design:

Selective encoding is regulated by **salience, novelty, affect/arousal, and goal value**, with neuromodulatory systems (e.g., noradrenergic locus coeruleus; dopaminergic novelty/reward loops; amygdala modulation) shaping what stabilizes into long-term memory. citeturn5search2turn5search4turn4search2turn4search3turn5search5

Memory persistence depends on **time-dependent plasticity** and “credit assignment” processes such as **synaptic tagging and capture** and its behavioural analogs: weak learning can become long-lasting when paired (within a window) with novelty/arousal that supplies consolidation resources. citeturn4search0turn4search5turn4search13turn4search1

Consolidation is not only synaptic; at the systems level, memories reorganize over time (hippocampus–neocortex interactions), increasingly supported by generalized representations and schemas. Sleep supports this with coordinated rhythms and replay (slow oscillations, spindles, hippocampal ripples), and also with homeostatic down-selection of synaptic strength. citeturn0search1turn2search4turn3search4turn3search8turn3search17turn3search0

Retrieval is not a read-only operation. Reactivation can render a memory **labile** and require **reconsolidation**, creating an opportunity for updating (and also a risk of distortion). citeturn0search2turn2search15turn2search23turn2search8

Recall is cue-driven: encoding specificity, context dependence, and transfer-appropriate processing show that “what is stored” includes **how it was processed** and **the state/context** at encoding, which determines what cues will later work. citeturn1search0turn1search1turn1search7

The hippocampal system supports **pattern separation** (reducing interference between similar episodes) and **pattern completion** (recovering a whole memory from partial cues), often formalized computationally as attractor-like dynamics. citeturn8search0turn8search1turn8search3turn8search13

Forgetting is not only decay: retrieval can induce forgetting of competitors (inhibitory control), and sleep may renormalize synaptic strength (“downscaling”), implying that a useful memory system must implement **active interference management** and **resource rebalancing**, not just retention. citeturn5search23turn3search17turn3search1

Humans privilege unfinished intentions (Zeigarnik effect) and rely on specialized mechanisms for prospective memory—remembering to act when a cue occurs (event-based) or when time passes (time-based). citeturn6search0turn6search2turn6search19turn6search12

Finally, memory is integrated with self-models and goals: self-referential encoding improves memory, is linked to medial prefrontal regions, and the default mode network participates in autobiographical memory and future-oriented simulation. citeturn10search16turn10search2turn10search9turn9search3turn7search10

## Mechanism-to-architecture proposals with implementable algorithms

Below, each requested dimension is paired with at least one **concrete retrieval architecture/algorithm**. All are compatible with a Markdown-first store: Markdown files hold canonical memory objects, while indices/graphs are **derived artefacts** that can be rebuilt nightly (no dependence on relational schemas; no dependence on embeddings being “the solution”).

### Memory encoding

**Brain mechanism.** Encoding is gated: novelty, arousal, and motivational relevance bias what persists, consistent with neuromodulatory models (LC-NE; hippocampal–VTA novelty loop; amygdala modulation) and arousal-biased competition. citeturn5search2turn5search4turn4search2turn4search3turn5search18

**Proposed architecture: Tag-and-Capture Encoding Gate (TCEG).**  
This mirrors synaptic tagging/capture and behavioural tagging: create many cheap “tags” now; allocate expensive consolidation resources later only for tagged items that are “captured” by novelty/arousal/goal events within a window. citeturn4search0turn4search5turn4search13turn4search1

**Data structures (Markdown + derived indices).**
- `/inbox/stream/YYYY-MM-DD.md`: append-only raw interaction/event stream (low friction).
- `/candidates/YYYY-MM-DD/*.md`: short “candidate memories” with YAML:
  - `tags: [people:X, project:Y, emotion:stress, novelty:high, goal:G1]`
  - `tag_strength ∈ [0,1]`, `capture_deadline` (e.g., +6h / +24h)
- Derived index: `/_index/tag_queue.json` (priority queue keyed by `capture_deadline`, `tag_strength`).

**Triggers.**
- **Tag trigger (cheap, frequent):** any event boundary, decision, conflict, or explicit “remember this” phrase.
- **Capture trigger (rarer, valuable):** novelty spike, emotional spike, goal-relevant milestone, surprising outcome (prediction error), or explicit reflection/review. citeturn4search3turn5search4turn4search2turn5search9

**Scoring function.**
Let `T` be tag strength (quick estimate), `N` novelty, `A` arousal, `G` goal relevance, `PE` prediction error proxy, `R` redundancy penalty, `U` unresolved-loop tension contribution.

- Tag score (online):  
  `T = σ( wN·N + wG·G + wA·A + wPE·PE − wR·R + wU·U )`
- Capture credit (on capture trigger): allocate a limited budget `B`:
  - promote candidate i to durable memory if `T_i · CaptureCredit ≥ θ` and the candidate is within its capture window.

**Retrieval flow impact.**
During retrieval, the assistant can also surface **recent high-tag candidates** even before they’re fully consolidated (like fragile “recent memory”). citeturn0search1turn2search4

**Tradeoffs and failure modes.**
- If `A`/arousal is over-weighted, the system may over-store emotionally intense but low-utility events (amygdala-like bias). citeturn4search2  
- If `R`/redundancy is aggressive, you risk losing weak signals that later become important (mis-set capture windows).
- “Novelty leakage”: random novelty (e.g., weird news) triggers capture of irrelevant items.

**Evaluation metrics.**
- **Encoding precision:** fraction of stored durable memories later used in responses or planning.
- **Miss rate under delayed relevance:** retrospective user queries (“Remember when…”) that fail because an event wasn’t captured.
- **Budget stability:** percentage of capture budget consumed per day; variance over time.

### Consolidation and reconsolidation

**Brain mechanism.** Reactivation can destabilize memory and require reconsolidation; consolidation also involves transformation and reorganization over time. citeturn0search2turn2search15turn2search8turn2search0

**Proposed architecture: Reconsolidation-as-Transactional Editing (RTE).**  
Treat every retrieval as opening a “labilization transaction” on the memory note: updates are staged, checked, and only committed during consolidation—preventing uncontrolled drift while still permitting updating like reconsolidation.

**Data structures.**
- Canonical memory objects: `/memories/*.md` with:
  - `claims:` (atomic statements)
  - `evidence:` links to source events (files + line anchors)
  - `confidence:` per claim
  - `last_recalled:`
- Reconsolidation staging: `/reconsolidation/queue/*.patch.md`
  - a patch contains proposed claim edits, plus why (new evidence, contradiction, outcome).
- Consistency ledger: `/_index/claim_graph.json` (directed graph: claim → supports/conflicts other claims).

**Triggers.**
- Any time a memory note is used to answer a question.
- Any time conflicting evidence appears (new event contradicts a stable claim).
- Any time the assistant makes a prediction and later observes the outcome (error-driven update). citeturn5search9turn2search23

**Scoring / commit rule.**
Define `ΔUtility` (expected future usefulness), `ΔConsistency` (conflict reduction), and `EvidenceStrength`.

Commit patch if:
`EvidenceStrength · (α·ΔUtility + β·ΔConsistency) − γ·DriftRisk > 0`

Drift risk rises if:
- patch changes a low-entropy (“core identity”) claim
- patch is based on single weak evidence
- patch conflicts with multiple higher-confidence claims

**Retrieval flow.**
1. Retrieve memory note(s).
2. Mark them `labilized=true` for a time window (e.g., 2 hours).
3. During the conversation, record new evidence and proposed updates as patches.
4. At 2 a.m., run consistency resolution and either commit or reject patches.

**Tradeoffs and failure modes.**
- Overly strict commit rules cause the system to feel “stubborn” (won’t learn).
- Overly permissive rules cause memory drift / confabulation, a known risk when recall modifies stored representations. citeturn2search8turn2search15  
- Patch conflicts can deadlock if the system lacks a principled tie-breaker (you need “goal/identity priors” and evidence provenance).

**Evaluation metrics.**
- **Drift rate:** proportion of claims changed without strong evidence.
- **Correction latency:** time from contradiction emergence to claim correction.
- **Stability/Plasticity balance:** fraction of updates that later revert.

### Hierarchical compression

**Brain mechanism.** Event perception and memory are hierarchical; consolidation transforms and compresses experience into schemas/gist while details may fade or become context-dependent. citeturn7search1turn7search12turn2search0turn2search5turn2search18

**Proposed architecture: Dual-Trace Hierarchical Compressor (DTHC).**  
Implement fuzzy-trace-like dual storage: keep (a) a compact **gist** and (b) a sparse set of **verbatim anchors** per timescale. citeturn2search6turn2search18

**Data structures.**
- `/events/YYYY/MM/DD/*.md`: segmented events (see embodied section for boundary detection).
- `/summaries/daily/YYYY-MM-DD.md`: day gist.
- `/summaries/weekly/YYYY-Www.md`: week gist + themes.
- `/schemas/*.md`: extracted general rules (“When X, Y tends to work”) linked to supporting episodes.
- Each summary stores:
  - `gist:` 5–15 lines
  - `anchors:` links to 3–10 key event blocks (quotes, decisions, outcomes)
  - `exceptions:` pointers to episodes that violated schema

**Triggers.**
- Nightly 2 a.m. consolidation (primary).
- Immediate after major “event boundary” clusters (optional micro-compression). citeturn7search1turn7search9

**Compression algorithm (concrete).**
1. **Segment** the day into events using boundary detection (details in embodied section). citeturn7search1turn7search12  
2. For each segment, compute a *segment signature* as a multiset of:
   - entities (people/projects)
   - action verbs
   - outcomes
   - affect/arousal tags
   - open-loop IDs referenced
3. Cluster segments by signature overlap to form “themes.”
4. For each theme, write:
   - gist sentence(s) = “what changed?” + “why it mattered?” + “what remains unresolved?”
5. Select anchors via a utility score:
   `AnchorScore = GoalImpact + Surprise + DecisionIrreversibility + SocialCommitment + EmotionalWeight` citeturn4search2turn5search4turn6search5  
6. Update `/schemas` by extracting repeated conditional patterns with exceptions (“schema + novelty” logic). citeturn2search5turn2search1  

**Retrieval flow implication (gist vs detail chooser).**
At recall time:
- If the user asks “What’s the overall situation with X?”, retrieve **schemas + weekly/daily gist** first.
- If the user asks “What exactly did I say / decide?”, retrieve **anchors + original event block**.

**Tradeoffs and failure modes.**
- Over-compression yields confident but wrong summaries (loss of anchors).
- Under-compression yields bloated context and slow recall (too many anchors).
- Schema overgeneralization can create “false regularities,” a known risk when gist dominates. citeturn2search18turn2search0

**Evaluation metrics.**
- **Compression ratio:** raw tokens/day vs daily summary tokens.
- **Answer faithfulness:** fraction of summary statements supported by anchors.
- **Schema utility:** how often a schema helps planning or reduces repeated mistakes.

### Salience and attention gating

**Brain mechanism.** Arousal can amplify competition: high-priority representations get stronger, low-priority ones get weaker (“winner-take-more/loser-take-less”), and emotional arousal modulates consolidation via amygdala-mediated interactions with other memory systems. citeturn4search3turn4search2

**Proposed architecture: Arousal-Gain Competitive Retrieval (AGCR).**  
Instead of “retrieve top-k,” perform **within-set competition** among candidates that share cues, with arousal/importance acting as a gain on competition (mirroring arousal-biased competition). citeturn4search3

**Data structures.**
- `/_index/competition_sets.json`: maps a cue (e.g., `project:Atlas`) to a set of candidate memory IDs that commonly compete.
- Each memory note stores `salience_base` and `salience_contextual`.

**Trigger types.**
- During retrieval whenever:
  - multiple candidates share a key cue
  - the user’s affect is elevated (stress/urgency proxy)
  - the query is high stakes (money/relationships/health)

**Scoring function (explicit).**
For candidates `i` in a competition set `C`:

1. Base evidence score by cue overlap and state match (from later sections): `E_i`.
2. Compute gain `g = 1 + k·Arousal`.
3. Competitive selection probability:
   `P(i) = softmax( g·(E_i + λ·Salience_i) )`

Then, apply **loser suppression** (see forgetting section) to reduce persistent intrusions.

**Retrieval flow.**
- First, assemble a candidate pool using symbolic cues (not vectors).
- Identify competition sets involved.
- Choose winners under the gain model; suppress losers mildly.

**Tradeoffs and failure modes.**
- Under high arousal, the system may become overly narrow (good for focus, bad for creativity).
- Over-suppression risks hiding relevant alternatives, especially for ambiguous queries.

**Evaluation metrics.**
- **Intrusion rate:** how often irrelevant but salient memories appear.
- **Focus gain under urgency:** reduction in time-to-relevant-answer for urgent queries.
- **Diversity loss:** decrease in alternative suggestions when arousal is high.

### State-dependent retrieval

**Brain mechanism.** Recall depends on overlap between encoding and retrieval contexts (encoding specificity; context-dependent memory), and on overlap between processing at encoding and processing at test (transfer-appropriate processing). citeturn1search0turn1search1turn1search7

**Proposed architecture: State-Indexed Cue Bundles (SICB).**  
Represent each memory with a compact “cue bundle” containing **state features** and **processing mode** features, then retrieve by weighted overlap rather than semantic proximity.

**Data structures.**
Each memory `.md` has YAML front matter like:

- `state:`
  - `time_of_day: morning|afternoon|night`
  - `location: home|office|travel|unknown`
  - `device_context: phone|laptop|car`
  - `social_context: alone|team|family`
  - `physio_proxy: low_energy|normal|high_arousal`
- `processing_mode: planning|reflection|argument|execution|learning`
- `cues: [entity:X, project:Y, action:buy, emotion:anxiety, goal:G]`

Derived indices:
- `/_index/inverted_cues.json`: cue → memory IDs
- `/_index/state_buckets.json`: discretized state → memory IDs

**Triggers.**
- Any retrieval request.
- Implicit: a state transition (e.g., arriving at work) can prefetch relevant open loops.

**Scoring function.**
Let `CueOverlap` = weighted Jaccard over cues, `StateMatch` = weighted match over state fields, and `ProcMatch` = indicator of processing mode alignment.

`Score = a·CueOverlap + b·StateMatch + c·ProcMatch − d·RecencyPenalty + e·GoalRelevance`

This directly instantiates cue-overlap principles rather than relying on dense semantic similarity alone. citeturn1search0turn1search7

**Retrieval flow.**
1. Infer current state and intended processing mode from the user’s request.
2. Pull candidates from inverted cue index.
3. Re-rank by `Score`.
4. If top candidates are low-confidence, request/derive additional cues (explicit “reminder cueing” like Tulving-style cue reinstatement). citeturn1search0

**Tradeoffs and failure modes.**
- Requires reliable state inference; errors cause “wrong-context recall.”
- User privacy: storing location/physio proxies is sensitive; allow coarse buckets and opt-outs.

**Evaluation metrics.**
- **State congruency benefit:** improvement in recall success when state features match.
- **Wrong-context errors:** memories retrieved from mismatching states.
- **Cue efficiency:** number of cues needed to reach a confident retrieval.

### Associative graph traversal

**Brain mechanism.** The hippocampus can act as an index binding distributed cortical patterns; recall can proceed via pattern completion from partial cues, while pattern separation reduces interference between similar episodes. citeturn0search8turn0search0turn8search1turn8search0

**Proposed architecture: Hippocampal Index Graph with Completion/Separation (HIG-CS).**  
Build a sparse graph where episodes are **indices** over shared feature nodes, then retrieve via spreading activation (completion) plus anti-interference mechanisms (separation + inhibition).

**Data structures.**
- Nodes:
  - Episode nodes: `E:2026-04-06T14:03:...`
  - Feature nodes: `F:person:…`, `F:project:…`, `F:action:…`, `F:emotion:…`, `F:place:…`, `F:goal:…`
- Edges: `E —(w)→ F` and optionally `F —(w)→ F` for learned associations.
- Storage:
  - Markdown episodes live under `/events/...`
  - Graph adjacency list: `/_index/hig_adjacency.json`

**Triggers.**
- On each event boundary: add/strengthen edges.
- On each recall: run activation traversal + update weights.

**Activation traversal algorithm (explicit).**
Given a cue set `Q` (feature nodes) derived from the query/state:

1. Initialize activations: `a(f)=1` for `f∈Q`.
2. Propagate for `t=1..T`:
   - `a_next(v) = Σ_{u∈Nbr(v)} a(u)·w(u,v)·exp(-δ)`  
3. Select episode candidates where `a(E) > θ`.
4. **Pattern separation step:** if top episodes are too similar (high feature overlap), require distinguishing context features (time, location, participants) and re-run with those cues. citeturn8search0turn8search3  
5. **Inhibition (retrieval-induced forgetting):** for competitors strongly activated but not selected, apply small negative weight updates or reduce their “availability” for this cue context. citeturn5search23turn5search7

**Retrieval flow.**
- Query → cues → activate graph → shortlist episodes → re-rank with state/goal priors → choose anchors/details → answer.

**Tradeoffs and failure modes.**
- Graph can become noisy if feature extraction is sloppy (over-connected “hub” nodes).
- Inhibition can hide useful alternatives if cue sets are too broad (important to limit inhibition to within a competition set). citeturn5search23  
- Requires periodic graph “renormalization” (see nightly loop).

**Evaluation metrics.**
- **Interference rate:** incorrect recall when similar episodes compete.
- **Cue robustness:** success under partial cues (“I can’t remember the name, but…”).
- **Hub dominance:** fraction of retrievals dominated by top hub nodes (should be bounded).

### Prediction and simulation

**Brain mechanism.** The hippocampal system is engaged in imagining future events and can generate sequences biased toward goals; replay-like sequences have been linked to planning/navigation and memory consolidation. citeturn7search10turn3search3turn0search7turn3search4

**Proposed architecture: Experience Graph Replay-and-Rollout (EGRR).**  
Instead of retrieval-for-answering only, maintain an experience-derived transition graph and use it for *simulation* (counterfactuals, planning), with nightly replay selecting experiences to strengthen and compress.

**Data structures.**
- `State` representation: discrete feature bundle (project phase, constraints, stakeholders, budget, time pressure).
- `Action` representation: action verbs + tool contexts.
- Directed multigraph: `S --(action, outcome stats)--> S'` stored in `/_index/experience_graph.json`
- Values: per-goal utility estimates stored in `/goals/*.md` and mirrored in `/_index/value_table.json` (derived).

**Triggers.**
- User asks “What should I do?” / “What are my options?” / “What’s likely to happen if…”
- High uncertainty (low confidence), high stakes, or repeated past failures.

**Rollout algorithm (implementable, no embeddings).**
1. Map current situation into a state `S0` (via rule-based feature extraction).
2. Retrieve relevant subgraph around `S0` using graph neighbourhood overlap (not semantic distance).
3. Run bounded rollouts of depth `d`:
   - choose actions based on past success frequency and goal value
   - simulate outcomes using empirical transition probabilities + recentness weighting
4. Return:
   - recommended actions
   - “why” episodes (anchors supporting the transition probabilities)
5. Nightly: perform “replay” by sampling surprising or high-impact transitions and strengthening schemas/values. citeturn3search4turn3search3

**Tradeoffs and failure modes.**
- If state discretization is too coarse, simulations become generic; too fine, and the graph fragments.
- Risk of reinforcing bad habits if reward signals are misestimated (needs explicit “outcome logging”).

**Evaluation metrics.**
- **Decision utility uplift:** user-rated usefulness of plans vs baseline retrieval.
- **Calibration:** alignment between predicted and observed outcomes.
- **Explainability:** fraction of recommendations backed by specific anchored episodes.

### Unresolved open-loops

**Brain mechanism.** Unfinished tasks can remain cognitively accessible (Zeigarnik effect), and prospective memory supports acting on future intentions via event-based or time-based cues. citeturn6search0turn6search2turn6search19turn6search7

**Proposed architecture: Open-Loop Tension Register with If–Then hooks (OLTR-ITH).**  
Store “open loops” as persistent intention objects with explicit trigger conditions and closure tests, integrated into recall ranking and proactive reminders.

**Data structures.**
- `/open_loops/index.md`: list of active loops with IDs.
- `/open_loops/OL-####.md` with YAML:
  - `intent:`
  - `created_from:` episode link(s)
  - `trigger_event:` cue bundle (e.g., `when: person=X AND topic=budget`)
  - `trigger_time:` schedule or deadline
  - `next_action:` explicit, small step
  - `tension:` numeric
  - `closure_condition:` testable criterion

**Triggers.**
- Event-based: cue match in conversation or environment (calendar location, detected person).
- Time-based: periodic checks; deadlines; the 2 a.m. loop refresh. citeturn6search2turn6search19

**Scoring function.**
`LoopPriority = p·tension + q·deadline_urgency + r·goal_alignment − s·user_annoyance_risk`

Update tension:
- increases after interruptions or repeated deferrals (Zeigarnik-like persistence)
- decreases after meaningful progress or explicit “drop this” instruction citeturn6search0turn6search5

**Retrieval flow.**
- Every user interaction runs a lightweight “loop trigger check” before general memory recall.
- If triggered, the assistant injects:
  - the loop title
  - the next action
  - the minimal context anchors

**Tradeoffs and failure modes.**
- Over-triggering feels nagging; under-triggering defeats the purpose.
- Incorrect triggers cause social awkwardness (“you said we should…” at the wrong time).

**Evaluation metrics.**
- **Loop closure rate:** percent of open loops resolved within target time.
- **Trigger precision:** percent of reminders judged timely/relevant.
- **Interruption cost:** user annoyance proxy (dismissals, “not now”).

### Embodied and action-linked memory

**Brain mechanism.** Memory is shaped by how events are parsed (event segmentation) and can be stronger when tied to actions/enactment; events are organized hierarchically and boundaries influence memory structure. citeturn7search1turn7search12turn7search4turn7search16

**Proposed architecture: Action–Outcome Script Memory (AOSM).**  
Store recurring action patterns as scripts keyed by tools, contexts, and outcomes, enabling recall by “what you’re doing” rather than “what you said.”

**Data structures.**
- `/scripts/*.md` (one per recurring routine), with:
  - `action_ngrams:` sequences of verbs (“draft → revise → send”)
  - `tool_state:` apps/devices involved
  - `common_failure_modes:` and fixes
  - `best_context:` (time, location, energy)
  - `evidence_episodes:` anchors to episodes where the script worked/failed

**Boundary detection trigger (concrete).**
Compute a boundary score from the event stream:
`B = Δ(people) + Δ(location) + Δ(tool_context) + Δ(goal_focus) + surprise`
A boundary is declared if `B > θB`, consistent with the idea that segmentation responds to prediction error and context change. citeturn7search12turn5search9

**Retrieval flow.**
- When intent looks like execution (“How do I do X right now?”), the assistant:
  1. builds an **action cue bundle** from verbs + tools mentioned/active
  2. retrieves scripts by action-ngram overlap
  3. returns steps plus the most recent outcome-linked anchor episodes

**Tradeoffs and failure modes.**
- If scripts become too rigid, they won’t adapt to novel contexts.
- If action extraction is noisy, script matching fails; require a controlled verb taxonomy.

**Evaluation metrics.**
- **Time-to-completion improvement** on repeated tasks.
- **Script hit rate:** frequency script retrieval is used and leads to progress.
- **Boundary accuracy:** agreement between detected boundaries and user-perceived event shifts.

### Forgetting and decay

**Brain mechanism.** Forgetting reflects decay, interference, inhibition (retrieval-induced forgetting), and sleep-linked renormalization of synaptic strength (homeostasis). citeturn5search23turn8search26turn3search17turn3search1

**Proposed architecture: Utility-Gated Decay with Interference Control (UGD-IC).**  
Implement a memory strength model that decays, is reinforced by use, and is reduced by competitor inhibition—plus a nightly global downscale that preserves relative importance, analogous to homeostatic renormalization.

**Data structures.**
Each memory note stores:
- `strength ∈ [0,1]`
- `last_access`
- `competition_set_ids`
- `goal_links`

A daily derived file `/_index/strength_table.json` summarises strengths for fast access.

**Update rules (explicit).**
- On recall of memory `m` at time `t`:
  - `strength_m ← min(1, strength_m + η·(1 − strength_m))`
  - for competitors `c` in same competition set:
    - `strength_c ← max(0, strength_c − κ·strength_m)` (bounded inhibition) citeturn5search23turn5search19
- Continuous decay:
  - `strength ← strength · exp(-λ·Δt)` with λ tuned by category (people > facts > momentary chatter)
- Nightly downscale (2 a.m.):
  - `strength_i ← strength_i / (mean_strength + ε)` (renormalize), then clamp.

**Pruning/compression rule.**
If `strength < θ_prune` and the memory has been fully subsumed by a schema or summary anchor, delete or merge it into a higher-level gist note.

**Tradeoffs and failure modes.**
- Too much inhibition: “fragile access” to less-used but important memories.
- Too much decay: user perceives forgetfulness.
- Too little pruning: memory bloat, slower retrieval, more noise.

**Evaluation metrics.**
- **Noise robustness:** retrieval precision under large stored volume.
- **Catastrophic forgetting rate:** important items lost (user-reported).
- **Interference reduction:** fewer wrong-memory intrusions over time.

### Identity and goal integration

**Brain mechanism.** Self-referential encoding improves memory (robust meta-analytic support) and involves medial prefrontal regions; DMN contributes to autobiographical memory and future-oriented cognition, linking memory to identity and simulation. citeturn10search16turn10search2turn10search9turn9search3turn10search1

**Proposed architecture: Self–Goal Prior with Comparator Updates (SGP-C).**  
Treat identity and active goals as priors that bias encoding, retrieval, and reconsolidation. A “comparator” checks whether new experiences confirm, refine, or contradict current self/goal models (inspired by PFC–hippocampal control ideas, though implemented at the algorithmic level). citeturn9search6turn9search2turn9search18

**Data structures.**
- `/identity/identity.md`: stable values, roles, boundaries, “things I never want again,” and long arcs.
- `/goals/*.md`: goals with states (active/paused), success criteria, and weights.
- `/identity/beliefs/*.md`: atomic belief notes, each with evidence links and confidence.

**Triggers.**
- Explicit self-referential content (“I care about…”, “I’m the kind of person who…”).
- Major outcomes (success/failure) that inform identity/goal priors.
- Repeated patterns detected by nightly schema extraction. citeturn2search5turn2search0

**Scoring function (used in encoding and retrieval).**
`SelfRelevance = overlap(cues, identity_tags)`  
`GoalRelevance = max_g overlap(cues, goal_g_tags) · weight_g`  

Then:
- Encoding gate adds `+ wS·SelfRelevance + wG·GoalRelevance` (self-reference effect analogue). citeturn10search16turn10search2  
- Retrieval ranking uses identity/goals as priors, not just matches.

**Comparator-driven updates (reconsolidation integration).**
When an episode strongly contradicts a belief/goal expectation:
- create `/identity/conflicts/CF-####.md` with:
  - conflicting claim
  - episode anchors
  - proposed resolution questions
- These conflicts are reviewed in weekly/monthly consolidation (to avoid knee-jerk identity drift).

**Tradeoffs and failure modes.**
- Over-strong identity priors can cause confirmation bias: it “remembers what fits the story.”
- Under-strong priors yield a system that stores facts but doesn’t learn personhood.

**Evaluation metrics.**
- **Personalization utility:** user-rated helpfulness of advice derived from identity/goals.
- **Bias auditing:** measure whether retrieval systematically excludes disconfirming evidence.
- **Stability of identity notes:** rate of oscillation vs gradual refinement.

### Retrieval flow diagram

```mermaid
flowchart TD
  U[User utterance or signal] --> P[Parse intent + extract cues]
  P --> S[Infer current state + processing mode]
  S --> OL[Open-loop trigger check]
  OL -->|hit| R1[Return loop + next action + anchors]
  OL -->|miss| C[Build cue bundle (entities/actions/goals/emotion)]
  C --> G[HIG-CS graph activation + pattern completion]
  G --> K[State + goal re-rank (SICB + SGP-C)]
  K --> Q[Competitive gating (AGCR) + competitor inhibition]
  Q --> A[Select gist vs anchors (DTHC)]
  A --> X[Generate response context]
  X --> Y[Respond]
  Y --> Z[Mark recalled items labilized; queue reconsolidation patches]
```

## Markdown folder structure and nightly consolidation rules for a 2 a.m. loop

This structure keeps Markdown as the canonical store while enabling fast recall through derived artefacts.

**Recommended folder structure.**
- `/inbox/stream/` — append-only daily streams (raw, low ceremony)
- `/events/YYYY/MM/DD/` — segmented event notes (boundary-based)
- `/summaries/daily/`, `/summaries/weekly/`, `/summaries/monthly/` — hierarchical gist + anchors
- `/schemas/` — generalizations with exception lists
- `/open_loops/` — intention objects (prospective memory)
- `/scripts/` — action/outcome scripts
- `/identity/` and `/goals/` — priors and weights
- `/reconsolidation/queue/` — staged patches
- `/_index/` — derived artefacts (rebuildable): inverted cue index, state buckets, hippocampal index graph adjacency, competition sets, strength table, experience graph

**Nightly consolidation (2 a.m., America/Vancouver) — concrete rules.**
1. **Segment the day’s stream** into events by boundary score `B` (context/tool/goal shifts + surprise). citeturn7search12turn7search1turn5search9  
2. **Run the tag-and-capture gate:** promote candidates whose tags were captured by novelty/arousal/goal triggers within window; drop the rest or compress into daily gist only. citeturn4search0turn4search5turn5search4turn4search2  
3. **Update HIG-CS graph:** add edges for new episodes; renormalize hub weights; recompute competition sets for cues that gained many episodes. citeturn0search8turn8search0  
4. **Build daily summary** with dual traces:
   - write gist (what changed, what mattered, what remains open)
   - select anchors by impact/surprise/commitment
5. **Schema update:** extract candidate schemas from repeated patterns; attach exceptions; update confidence gradually (no big jumps). citeturn2search5turn2search1  
6. **Reconsolidation commit:** evaluate queued patches; merge if evidence/consistency thresholds pass; otherwise defer to weekly review. citeturn0search2turn2search15  
7. **Forgetting/downscaling pass:** apply decay + inhibition updates; renormalize strengths; prune subsumed low-strength episodes (keep anchors). citeturn3search17turn5search23turn8search26  
8. **Experience graph update:** extract state/action/outcome transitions and update planning statistics; flag surprising transitions for replay emphasis. citeturn3search4turn3search3  

### Consolidation flow diagram

```mermaid
flowchart LR
  S[Daily stream + candidates] --> B[Boundary detection -> events]
  B --> T[Tag-and-capture promotion]
  T --> E[Durable event notes]
  E --> H[Update HIG-CS + competition sets]
  E --> D[Daily dual-trace summary (gist + anchors)]
  D --> W[Weekly/monthly rollups when due]
  D --> SC[Schema extraction + exception tracking]
  E --> EG[Experience graph update]
  H --> F[Decay + inhibition + renormalize strengths]
  SC --> I[Identity/goal comparator updates]
  E --> R[Reconsolidation patch commit]
```

### Timeline chart of consolidation cycles

```mermaid
timeline
  title Consolidation cycles for assistant memory
  section Online (seconds to minutes)
    Tag candidate memories on boundaries : continuous
    Update open-loop tension on interruptions : continuous
    Micro-summary after a session ends : ad hoc
  section Nightly (02:00 local)
    Segment -> summarize -> schema update : every night
    Reconsolidation patch commit + strength renormalization : every night
    Experience-graph replay selection : every night
  section Weekly
    Merge themes into higher-level schemas : weekly
    Review unresolved identity conflicts : weekly
  section Monthly
    Prune or archive low-utility clusters : monthly
    Re-weight goals and long arcs : monthly
```

## Comparison table of proposed architectures

| Mechanism | Proposed architecture | Data structure | Trigger types | Retrieval latency | Storage cost | Robustness to noise | Ease of implementation |
|---|---|---|---|---|---|---|---|
| Encoding (selectivity) | Tag-and-Capture Encoding Gate (TCEG) | Candidate notes + tag queue index | event boundaries; novelty/arousal; goal events | Low (uses small candidate set) | Low–Medium | Medium (depends on tag quality) | Medium |
| Consolidation / reconsolidation | Reconsolidation-as-Transactional Editing (RTE) | Versioned `.md` + patch queue + claim graph | any recall; contradictions; outcome feedback | Medium | Medium | High (drift controls) | Medium–Hard |
| Hierarchical compression | Dual-Trace Hierarchical Compressor (DTHC) | Daily/weekly/monthly summaries + anchors + schemas | nightly; boundary clusters | Low (gist first) | Low | Medium–High | Medium |
| Salience gating | Arousal-Gain Competitive Retrieval (AGCR) | Competition sets + salience fields | ambiguity; urgency; high arousal | Low–Medium | Low | Medium (can over-focus) | Medium |
| State-dependent recall | State-Indexed Cue Bundles (SICB) | YAML state fields + state buckets + inverted cues | every query; state transitions | Low | Low–Medium | Medium (state inference errors) | Medium |
| Associative traversal | Hippocampal Index Graph (HIG-CS) | Feature–episode adjacency list | cue overlap; partial cues | Medium | Medium | High if renormalized | Hard |
| Prediction / simulation | Experience Graph Replay-and-Rollout (EGRR) | State–action–outcome multigraph + value table | planning prompts; uncertainty; high stakes | Medium–High | Medium | Medium (state discretization risk) | Hard |
| Open loops | Open-Loop Tension Register (OLTR-ITH) | Per-loop `.md` + trigger matcher | event cues; time cues; interruptions | Very Low | Low | High (if triggers precise) | Easy–Medium |
| Embodied/action-linked | Action–Outcome Script Memory (AOSM) | Script library + action n-grams + evidence anchors | execution intents; tool context | Low | Medium | Medium | Medium |
| Forgetting/decay | Utility-Gated Decay with Interference Control (UGD-IC) | Strength table + decay/inhibition rules | continuous; nightly downscale | Low (ranking feature) | Low | High (reduces bloat) | Medium |
| Identity/goal integration | Self–Goal Prior + Comparator (SGP-C) | Identity/goals notes + conflict files | self statements; outcomes; schema changes | Low–Medium | Low | Medium–High | Medium |

## Evaluation strategy and metrics across the full system

A brain-inspired memory stack should be evaluated not only by “can it retrieve something,” but by **cue sensitivity, interference control, stability under reconsolidation, and planning utility**—the very dimensions that differentiate human memory from a static database. citeturn1search0turn5search23turn2search8turn3search4turn6search2

A practical evaluation harness for a personal assistant can include:

- **Cue-based recall tests:** Generate query variants that differ in cues (names vs places vs tools vs mood/state) and measure retrieval success and required clarifications, aligned with encoding specificity and context dependence. citeturn1search0turn1search1  
- **Interference benchmarks:** Create clusters of near-duplicate episodes (same project/person, different dates/outcomes) and measure confusion rate before/after enabling pattern separation and inhibition (retrieval-induced forgetting analogue). citeturn8search0turn5search23  
- **Reconsolidation stability tests:** Repeatedly retrieve a memory, introduce new contradictory evidence, and measure whether updates converge (correctly) without uncontrolled drift. citeturn0search2turn2search15turn2search19  
- **Compression fidelity audits:** Sample daily/weekly summaries and verify every summary claim is supported by at least one anchor (faithfulness), while ensuring high-level questions are answered faster than searching raw events. citeturn2search18turn7search12  
- **Open-loop utility metrics:** Track loop closure time, reminder acceptance rate, and false-trigger rate; compare explicit if–then triggers (implementation-intention style) to generic reminders. citeturn6search7turn6search2turn6search0  
- **Planning uplift:** For repeated decision domains (purchases, hiring, health routines), compare outcomes when using experience-graph rollouts vs retrieval-only (user satisfaction, reduced repeated mistakes, better calibration). citeturn3search3turn7search10turn3search4  
- **Resource/bloat management:** Monitor growth of raw events vs summaries and the fraction of retrievals that rely on high-level gist (desired) vs scanning raw logs (undesired), consistent with the need for consolidation/transformation and down-selection. citeturn3search17turn2search0turn2search4  

Together, these evaluations test whether the assistant’s memory behaves like a human-relevant system: **selective encoding, state-sensitive recall, interference control, adaptive updating, and multi-scale compression in service of goals and identity**. citeturn5search4turn4search3turn1search0turn5search23turn10search16turn3search4