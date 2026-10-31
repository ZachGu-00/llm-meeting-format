# Cloud handoff instructions

Read `handoff/STATE.json`, `reports/PAUSE_SUMMARY.md`, and `docs/CLOUD_SETUP.md` first.

The user paused the experiment at E4. A new environment, a clone, or a status request does not authorize resuming. Only run the resume command after the user explicitly asks to continue in this session.

The GPU server is authoritative: `/workspace/voices_nonadditivity_v3`. This repository is a small control and evidence snapshot; it is not a replacement installation. Do not run inference in the Codex container, overwrite server code, reinstall CUDA, delete locks, or start a second experiment queue.

Run `python3 cloud/control.py status` first. If SSH fails, resolve connectivity/authentication before touching experiment state. Internet access in a hosted environment does not prove that outbound SSH works.

The five original processes are SIGSTOP-paused in an existing screen. `python3 cloud/control.py resume --confirm-resume` checks the exact frozen score plan and process identities before sending SIGCONT. If any process is missing, stop and inspect; do not recreate a queue blindly. This initial E4 score plan is incomplete, so the phase wrapper's `--score-plan` recovery option (which requires a closed scoring plan) is not currently applicable.

Keep native BF16/eager/batch16, frozen six models and six topics, model revisions, readout calibration, E1 kernels, and source hashes unchanged. Never ask models to output numeric probabilities. E3 was explicitly skipped. The affine position check is a diagnostic on twenty previously used items, not a fresh confirmation or all of E9.

Completion requires all 36 model-topic scores and CPU analyses, six model aggregates and supplements, H3, E4a/b six-round registered Monte Carlo, final report, and verified backups. Do not infer completion from process exit alone. Compare exact plans, terminal receipts and source bindings. Report negative or uncertain results faithfully; no effect-dependent exclusion.

`cloud/control.py fetch-reports` downloads bounded report/status snapshots and verifies their transport hashes. It is NOT a full raw-data backup. For complete closed-artifact backups, use the reviewed `experiment/setup/backup_e4_monitor_v3.py` with its companion helper after establishing a full dependency mirror, or reuse its remote closure/archive logic and verify every source and prior dependency. Prior Windows backups exist; their monitor was stopped for handoff.

Do not commit credentials, private keys, large inputs, raw inference outputs or model weights. The repository's `experiment/` files are exact source snapshots. Any future change must be separately reviewed and must not silently invalidate an existing frozen plan.

