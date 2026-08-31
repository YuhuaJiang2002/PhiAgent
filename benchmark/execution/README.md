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

## Real blind evaluation

[`../protocols/real-robot-blind-v0.1.json`](../protocols/real-robot-blind-v0.1.json)
freezes the initial real protocol. Simulation-passing requests are committed
before execution; operators and outcome reviewers are separated; reviewers do
not see method or seed; all attempts remain in the denominator. Hardware
execution remains disabled in checked-in adapters until a site-specific safety
review changes `execution_enabled` under an authorized deployment.
