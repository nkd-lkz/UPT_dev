# Separate contact-stage changes from drive changes

This diagnostic tests whether response-memory errors come from changed contact or changed arm stiffness. It follows the four-arm frozen baseline audit and the correct/no/matched-wrong-history probe. It does not train a controller or measure task success.

## Run the prerequisites first

The October 6 waiting campaign expired without completing GPU jobs. The October 8 campaign restarts those jobs in a new directory. It preserves baseline revision `7335f2f4`, diagnostic revision `f74d8db0`, checkpoint hashes, four seeds 4101–4104, and 16 episodes per seed. Each GPU has a 12-hour execution ceiling and a 36-hour waiting deadline. The ceiling is not a target: completed or invalid experiments stop.

BC-only, Q+BC, reference and the old zero head must finish all 256 episodes with frozen weights, correct routing and matched initial observations. Only BC-only versus Q+BC shares a fresh training budget. Then run 56 paired histories with identical current state and command. Train the equal-capacity response heads for 512 updates with three paired initialization/sampling seeds. These are diagnostic predictors, not Stage 2 actors.

## Cross two causes of response change

`toolkits.rlt.probe_memory_conditions` adds a `phase` mode. A contact stage is an observed interaction state, not the index of an attempt. Each seed has the following eight schedules:

| Contact schedule | Fixed stiffness | A→B→A stiffness |
|---|---|---|
| Free / free / free | 1000 / 1000 / 1000 | 1000 / 250 / 1000 |
| Grasp / grasp / grasp | 1000 / 1000 / 1000 | 1000 / 250 / 1000 |
| Free / grasp / free | 1000 / 1000 / 1000 | 1000 / 250 / 1000 |
| Grasp / free / grasp | 1000 / 1000 / 1000 | 1000 / 250 / 1000 |

Six seeds, 57001–57006, each generate twelve ten-tick commands. The same commands repeat in all three attempts and all eight schedules. Arm commands lie in ±0.02 normalized units; the gripper command is always −1. Each attempt uses a seeded scene reset. For grasping, initialization places the peg at the fingers and sets finger positions near its width; twenty physical settling ticks follow. This privileged construction isolates response behavior. It is not a demonstrated grasp, a recovery expert, or an evaluation of insertion.

The tool records both finger–peg force magnitudes and ManiSkill's grasp predicate after every control tick. A free interval requires zero grasp ticks and both forces below 0.1 N. A grasp interval requires the grasp predicate on at least 90% of ticks. The last ten settling ticks and each full attempt must pass. Failed preparation stops the job. Slipping or unintended contact invalidates the stage comparison; the queue cannot mark it valid. These thresholds are engineering acceptance rules, not calibrated confidence.

Clear, retain, time decay and consequence-error weighting read the same completed commands and displacements. The predictor receives no stage label, stiffness label, contact audit or future response. It predicts before observing each completed chunk. Report all four readers, per-seed full-attempt MSE and first-four-chunk MSE. Compare readers within each cell before comparing cells. Repeated scene resets and constructed grasps limit conclusions to response identification. The test does not establish episode memory retention in deployment.

## Execute and accept the artifact

On an idle GPU, with the project virtual environment and a new output path:

```bash
"$RLINF_VENV/bin/python" -m toolkits.rlt.probe_memory_conditions phase \
  --gpu 0 --stages fixed --smoke --output "$CAMPAIGN/phase_fixed_smoke"
"$RLINF_VENV/bin/python" -m toolkits.rlt.probe_memory_conditions phase \
  --gpu 1 --stages changing --smoke --output "$CAMPAIGN/phase_changing_smoke"
```

Only after each smoke passes, repeat its command without `--smoke` and with a new output path. Each full half contains 24 streams. The GPU lease and occupancy checks apply to both commands. `results.json` must report completion, the expected stream count and `all_contact_stages_valid=true`. Raw commands and outcomes are in `phase-streams.pt`; the report records its SHA-256. A smoke validates integration on one seed; it is not a research result.

CPU contract tests cover the factorial, contact rejection and queue validation. GPU integration remains pending until the queued smoke completes. A renderer-disabled CPU attempt failed inside the installed SAPIEN URDF material loader; it is not an alternative validated backend.

## Decide what the result permits

Correct history must outperform both absent and matched-wrong history on held-out pairs before claiming extra information. Error weighting must improve on simple retention under the intended changes before promoting it. The existing synthetic return-to-A counterexample remains: retention MSE 0.000389, error weighting 0.000696. Even positive prediction results require a matched closed-loop control test before Stage 1B or a formal applicability module is justified. No current result establishes reduced intervention or physical RSI.

The official RLinf main branch was checked on October 8: `0067f7d5` includes merged PR #1623, which fixes retained replay checkpoint contents. PR #1658, on π0.5 conversion/model-path documentation, remains open. The frozen campaign incorporates neither change. Validate replay save/load separately before resuming online training; the present evaluation restores model weights only.
