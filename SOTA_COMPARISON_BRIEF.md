# SOTA comparison: what is and isn't completed

This is a documented literature comparison and benchmark plan, **not** a numerical SOTA replication.

| Reference | What the published method does | Comparison to our implementation | Status |
|---|---|---|---|
| ADVICE — Bethell et al., ECAI 2025 | Learns a contrastive-autoencoder safety classifier for black-box environments; substitutes a high-critic-value safe action | Our explicit GridWorld safe-action oracle assumes known local safety; our Count-Aware FABR chooses among safe actions with Q, recent unsafe proposals and state-action visitation novelty. Our value-based baseline resembles the value-selection step, **not a reproduction of the ADVICE shield**. | Literature comparison only |
| tempestpy — Pranger & Könighofer, 2026 | Formal shielding integrated with Gymnasium/Tempest; MiniGridSafe; MaskablePPO | Formal guarantees and stochastic environments differ from our deterministic post-shield and DQN. Our percentage on a different maze is not numerically comparable. | Shared-benchmark replication not run |
| Dynamic Safety Shield — Dawood et al., L4DC 2025 | Dynamic shielding intended for robot navigation | Motivates learning-aware intervention; different control system, safety model, and benchmarks | Literature comparison only |

## A fair numerical SOTA experiment must include

The same scenario distribution, training timesteps, deployment safety constraints, metric definitions, and shield-on versus shield-off conditions. A jointly instrumented benchmark must include both the published reference and our implementation; if reproducing the paper's code is not feasible in time, explain why and leave numerical cells blank. Sai owns this verification; do not invent numbers.

Sources: https://arxiv.org/abs/2405.18180 ; https://arxiv.org/abs/2606.03804 ; https://proceedings.mlr.press/v283/dawood25a.html
