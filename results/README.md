# Published numerical results

- `baseline_75/` contains the 15 estimator, 15 controller-comparison and 45
  robustness cases.
- `applicability_99/` contains the fixed 36-case screening, 18-case bandwidth
  study and 45-case independent confirmation.
- `reference/` contains the complete public input and compact output for
  `mixed_holdout_103` in MPC (`ignore`), EnvMPC (`estimate`) and true-wrench
  (`oracle`) modes.

Large Python snapshots are intentionally omitted. Run
`python scripts/verify_results.py` to check cardinalities, sanitization and
the SHA-256 inventory in `CHECKSUMS.sha256`, or
`./scripts/run_reference.sh` to regenerate the reference case.
