# Simulator and real-robot execution plan

## RoboWM-Bench / Isaac Lab 5.1

Use a dedicated Python 3.11 environment matching
[`robowm-isaaclab5.1.json`](robowm-isaaclab5.1.json). First run the read-only
preflight:

```bash
python -m phiagent.benchmark.cli robowm-preflight \
  --checkout external/RoboWM-Bench \
  --revision 0a8b0eab3ebfb7993f6ab895f12eac41dfefa1c1
```

Then emit one episode command with `robowm-command`. Each measured episode must
use a new output directory and preserve failures. The upstream task-success log
is imported only as task outcome. PhiAgent must separately record collisions,
joint and velocity violations, minimum Jacobian singular value, contact-stage
success, GPU identity, revisions, and timings before `physical_gate_complete`
may become true.

The pinned checkout needs the three small compatibility patches under
[`patches/`](patches/): LeRobot is imported only when dataset export is
requested, hardware teleoperation is optional during offline replay, and a task
without `get_part_scores()` no longer converts an otherwise valid replay into a
false failure. Apply them in filename order and verify them with
`git apply --check` first. They do not modify dynamics, actions, success checks,
or task assets.

For pip-installed Isaac Sim 5.1, set `OMNI_KIT_ACCEPT_EULA=YES`; `ACCEPT_EULA`
is a container-image variable and does not dismiss the pip-runtime prompt. The
headless command also requires `--enable_cameras`, because RoboWM task
observations contain tiled cameras. `PYNPUT_BACKEND=dummy` prevents an X-server
dependency on offline workers.

The 2026-09-01 pilot ran the frozen `Franka-pick` episode 0 on one isolated RTX
PRO 5000 72GB Blackwell and obtained `1/1` upstream task success. Its episode
and pose hashes are frozen in
[`robowm-isaaclab5.1.json`](robowm-isaaclab5.1.json). This remains a partial L4
integration: the published scene emits a missing NuRec field warning and a
PhysX contact-filter cardinality warning, and the replay script does not export
all PhiAgent collision, joint-limit, velocity, singularity, and contact gates.
The upstream success is therefore stored as `task_outcome_only`, never as
`physical_gate_complete=true`.

## Real blind evaluation

[`../protocols/real-robot-blind-v0.1.json`](../protocols/real-robot-blind-v0.1.json)
freezes the initial real protocol. Simulation-passing requests are committed
before execution; operators and outcome reviewers are separated; reviewers do
not see method or seed; all attempts remain in the denominator. Hardware
execution remains disabled in checked-in adapters until a site-specific safety
review changes `execution_enabled` under an authorized deployment.

Every trial must additionally include a pre-registration manifest containing
the protocol/trial/case identifiers, a timezone-aware registration timestamp,
and hashes of the action, calibration, initial-state video, and predicted video.
`real-trial-check` rejects any post-registration mutation and hashes the full
nine-artifact bundle without invoking hardware:

```bash
python -m phiagent.benchmark.cli real-trial-check \
  --descriptor /recorded/trial/trial.json \
  --adapter-manifest benchmark/adapters/rm65-ag2f90d-recorded.json \
  --protocol benchmark/protocols/real-robot-blind-v0.1.json \
  --session-id blind-session-001 --trial-index 0 \
  --reviewer-id-hash <lowercase-sha256> \
  --output /recorded/trial/validated-evidence.json
```
