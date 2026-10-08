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

## Continue with an input-scaling audit and closed-loop tracking

All 24 baseline/response/contact jobs finished by 20:39 on October 8. Frozen control results are BC-only 18/64, Q+BC 19/64, reference 21/64 and old zero 21/64; all have 26/64 episodes reaching the gate. Correct-history fixed-response test MSE is 2.89e-7 versus 9.68e-5 for a training-set global gain and 3.82e-4 for wrong history. The neural head stays near 4.8e-4 with or without history. Error weighting does not consistently beat retention across contact/drive schedules. These results motivate diagnosing how useful evidence reaches a decision before adding Stage 1B.

`fit --standardize` in `toolkits.rlt.probe_memory_conditions` reuses the original matched dataset and its train/validation/test partition. It fits input means and scales only on training pairs, then uses the same architecture, three seeds, 512 updates and sample sequences. Disabled history is zero after normalization; targets, pair IDs and condition labels remain excluded. Because these data have already been inspected, this is a post-hoc optimization diagnosis, not a new held-out validation claim.

`toolkits.rlt.probe_response_control` tests whether the existing fixed response statistics can improve a simple feedback controller. It does not replace or train RLT's actor. A seven-joint sinusoidal target path repeats in all three attempts. Every method gets identical initial-state hashes and goals. Five methods use the same previously fitted global gain: fixed, clear, retain, decay and error weighting. Adaptive gains are ridge-regularized toward that prior, clipped to [0.25, 2], and used to divide the current goal error. Commands are clipped to ±0.08 and executed for ten ticks. Only the executed command and completed displacement enter memory. Contact and stiffness labels stay outside the controller.

The eight contact×drive schedules are unchanged. Smoke uses seed 58001; the full study uses new seeds 58101–58106. Each method executes 24 chunks and 20 setup ticks per attempt, or 780 control ticks per three-attempt stream. Fixed/changing halves each contain 120 streams and 93,600 ticks. Report per-seed tracking MSE, first-four-chunk error, 5 mrad goal attainment, clipping, command energy and contact failures. Keep failed-contact streams in the report; do not select successful grasps to improve a method's average. Lower tracking error with more grasp failures does not establish a contact-control improvement. The preserved initial-state construction limits conclusions to local joint tracking, not insertion success, correction savings or physical RSI.

```bash
CUDA_VISIBLE_DEVICES='' "$RLINF_VENV/bin/python" -m toolkits.rlt.probe_memory_conditions fit \
  --standardize --dataset "$MATCHED_DATA" --output "$CAMPAIGN/scaled_fit"
"$RLINF_VENV/bin/python" -m toolkits.rlt.probe_response_control \
  --gpu 0 --stages fixed --smoke --dataset "$MATCHED_DATA" --output "$CAMPAIGN/fixed_smoke"
```

Use `--gpu 1 --stages changing` for the second smoke. After engineering acceptance, omit `--smoke` and use a new output path. `MATCHED_DATA` is the original `matched.pt`; its hash is recorded. A positive tracking result would justify a separate, gated RLT action-adapter experiment, not an immediate claim about the trained actor.
