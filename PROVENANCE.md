# Provenance

This repository is a history-free publication snapshot prepared from the BELKA
development repositories. It intentionally contains no robot runtime, stand
software, flight records or payload-identification code.

| Component | Source revision | Publication treatment |
|---|---|---|
| HabitatSim experiment workspace | `Belka/habitat_sim@e777e1928d6451f78a2e3b984dc7ca5762122021` | Only the MuJoCo plant and external-wrench experiment were retained. |
| Robot model, MPC and ESKF | `Belka/control@6d4027eb0714c0c67caa0922cf60ab8355b61d1f` | Vendored as the ordinary `belka/` package. This is the campaign version with truth references and the ESKF covariance-reset Jacobian. |
| Tracking state machine | `Belka/scenarios@5cca9ea82ec1657a927fce018530d6d67c34dca1` | Only `tracking.py` was vendored; no private repository is required. |
| acados | `acados/acados@8e1a6f856e063c423de583e03c691e3b2b7fc0a0` | Public pinned submodule; the default HPIPM backend is used. |

The complete archived results record `2960bca2a7d0ed66303f42a9a8951ee759ff9cb8`
as the superproject HEAD present during the numerical campaign. The corrected
control submodule was used from `6d4027e`; its pointer had not yet been recorded
by that superproject commit. The publication snapshot retains the final
experiment implementation from `habitat_sim@e777e192`. The compact result
bundle was copied from the article archive with machine-local paths and omitted
`snapshot.pkl` references removed. Physical metrics were not recomputed during
that copy.

The original BELKA acados mirror contains two subsequent commits that change
optional qpOASES/Clarabel submodule pins and internal mirror URLs. Neither
change affects the HPIPM configuration used here, so the public upstream parent
commit is sufficient for the published controller.
