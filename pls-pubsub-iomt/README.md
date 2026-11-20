# PLS-Aware Topic Orchestration for IoMT Publish/Subscribe Systems — experiment code

Three evaluation layers, matching the experiment design doc:

| Layer | What it does | Where it runs | Entry point |
|---|---|---|---|
| L1 PHY | AN-aided multicast secrecy: analysis vs Monte Carlo, optimal φ, SOP lookup, link traces | Colab | `python -m l1_phy.run_l1` |
| L2 DRL | Dueling Double DQN orchestrator, baselines B1–B5, ablations, β_s sweep | Colab (GPU) | `python -m l2_drl.run_l2_all` |
| L3 testbed | Containernet + Mosquitto + Ryu, trace-driven netem, crypto benchmark | WSL2 / Ubuntu 24.04 | `l3_testbed/run_all_l3.py` |

All parameters live in `config/params.yaml` and `config/topics.yaml`. Every run writes to
`results/<layer>/<name>_<timestamp>[_s<seed>]/` with a `meta.json` (seed, git hash, config).

## Order of work

1. **Colab — smoke test.** Upload this folder to My Drive, open `notebooks/colab_runner.ipynb`, run section 1.
2. **Colab — L1 full.** Section 2 (set `mc_trials: 1000000` for the final figures).
3. **Colab — L2 training and evaluation.** Sections 3–4. Jobs are independent; split them across sessions with `--variants` / `--seeds`.
4. **Copy `results/l1` and `results/l2` from Drive** into the repo on your WSL machine.
5. **Testbed setup** (once): `bash l3_testbed/setup_host.sh`, then start Ryu and run the smoke test:
   ```
   sudo docker run -d --rm --name iomt-ryu --network host iomt-ryu
   sudo -E ~/cn-venv/bin/python l3_testbed/smoke_test.py
   ```
6. **Testbed runs:** quick check, then the full plan (crypto benchmark + 5 schemes × 5 repetitions):
   ```
   sudo -E ~/cn-venv/bin/python l3_testbed/run_all_l3.py --reps 1 --duration 60 --warmup 10
   sudo -E ~/cn-venv/bin/python l3_testbed/run_all_l3.py
   ```
7. **Figures:** `python figures/plot_all.py` → `figures/F1…F12.pdf/png`. Missing inputs are skipped.

## Layout

```
config/        params.yaml, topics.yaml
common.py      config loading, path loss, result folders
l1_phy/        channel.py (Monte Carlo), sop_analytic.py (analysis), traces.py, run_l1.py
l2_drl/        env.py, dueling_agent.py, baselines.py, policy_np.py, train.py, evaluate.py, run_l2_all.py
l3_testbed/    setup_host.sh, docker/, run_experiment.py, run_all_l3.py, smoke_test.py,
               publisher.py, subscriber.py, orchestrator.py, replay_traces.py, crypto_bench.py,
               ryu_app.py, analyze_l3.py, gen_certs.sh
notebooks/     colab_runner.ipynb
figures/       plot_all.py
```

## Model notes (for the paper's system-model section)

- **SOP analysis.** Eavesdropper statistics are exact for any K (w ⟂ AN space). K = 1 is fully
  analytical; K > 1 samples the weakest-subscriber SNR and keeps the eavesdropper side analytical
  ("semi-analytical"). Both match Monte Carlo (F1–F3).
- **Operating point.** R_s = 2 and R_min = 5 bit/s/Hz at 20 dBm were chosen because they give a real
  trade-off: lower φ cuts SOP but raises decoding outage, and 3-subscriber topics need splitting to reach ε = 0.01.
  Justify or change these before the final runs; everything downstream follows the config.
- **L2 decisions.** One decision per topic per 1 s epoch, in priority order: tier × grouping × φ (20 actions).
  B4 greedy is a one-step model-based oracle with exact lookup access, so it is a strong baseline.
- **Testbed simplifications.** The PHY is not emulated: L1 traces drive netem on each subscriber link
  (keyed by the subscriber's highest-priority topic). The orchestrator builds its observation from a
  digital twin (the L2 environment with the testbed's fixed distances). The uplink has no PLS (static
  3 ms netem, 2 Mbit/s). Mosquitto is far faster than `edge_service_rate`; to emulate MEC contention,
  cap the edge broker with `--edge-cpu` and calibrate it to the configured service rate.

## What has been tested

L1 (quick mode), the L2 environment, all baselines, evaluation and NumPy inference, the orchestrator,
trace replay, log analysis and the figures were run end to end. The PyTorch training loop
(`dueling_agent.py`, `train.py`) and anything that needs Containernet, Docker, paho-mqtt or Ryu
were syntax-checked only; run the smoke tests first and report any error.
