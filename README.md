# Adversarial Reinforcement Learning for Emotion-Fake Audio Detection

## Capstone Execution Blueprint

This repository follows a **5-phase plan** for emotion deepfake detection with adversarial RL:

- **Phase 0**: Setup (environment, data integrity, preprocessing, PESQ/STOI validation)
- **Phase 1**: Supervised baselines (defender SFT, attacker pretraining)
- **Phase 2**: Cross-architecture stress testing
- **Phase 3**: 5,000-step zero-sum RL loop (GRPO + curriculum + WGAN-GP)
- **Phase 4**: Final evaluation, zero-day transfer, statistical validation

---

## 1) Model Selection and Role Assignment

### Defenders

- **WavLM-Base (94M backbone + trainable head only)**  
  Role: strongest baseline + RL participant (active defender in alternation schedule).
- **AASIST (graph attention spoof detector)**  
  Role: attacker-SFT target (frozen in Phase 1B) + RL participant.
- **RawNet2 (sinc-conv waveform model)**  
  Role: evaluation witness (frozen for transfer/generalization checks).
- **LCNN (LFCC-based lightweight CNN)**  
  Role: evaluation witness and lower-bound robustness reference.

### Attackers

- **PGD**: baseline white-box attacker (Phase 2 ceiling reference).
- **FGSM**: sanity attacker (Phase 4 quick robustness check).
- **CNN U-Net**: spectrogram-space cross-arch evaluation attacker.
- **Bi-LSTM**: temporal-transition cross-arch evaluation attacker.
- **Squeezeformer**: primary RL attacker (steps 1–3000).
- **Mamba**: secondary RL attacker (steps 3001–5000; long-sequence efficient).

### RL loop pairing

- **Active defenders in RL**: WavLM-Base + AASIST (alternate every 100 steps)
- **Active attackers in RL**: Squeezeformer (1–3000), then Mamba (3001–5000)

---

## 2) Phase-by-Phase Plan

### Phase 0 — Setup

1. Pin environment (PyTorch 2.x + CUDA 12.x, torchaudio, transformers, speechbrain, pesq, pystoi, scipy, wandb).
2. Download EmoFake and inspect metadata.
3. Create fixed **speaker-disjoint** splits with seed 42:
   - train 70%
   - dev 10%
   - test 20% (locked until Phase 4)
4. Standardize preprocessing (16 kHz mono, peak-normalize to [-1, 1], pad/trim to 4s).
5. Cache LFCC offline for LCNN.
6. Validate PESQ/STOI pipeline on sample pairs.

Exit criteria:

- PESQ/STOI values in valid ranges
- split manifests saved and fixed
- consistent preprocessed outputs across defender data loaders

Phase 0 starter implementation in this repository:

- `environment.phase0.yml`: pinned CUDA 12.1 + PyTorch 2.4 environment
- `scripts/phase0/build_splits.py`: metadata integrity checks + speaker-disjoint split manifests
- `scripts/phase0/preprocess_audio.py`: 16 kHz mono + peak normalization + pad/trim to 4s
- `scripts/phase0/validate_pesq_stoi.py`: PESQ/STOI validation for clean/degraded pairs

Example usage:

1. Create environment:
   - `conda env create -f environment.phase0.yml`
2. Build fixed manifests:
   - `python scripts/phase0/build_splits.py --metadata /path/to/metadata.csv --audio-root /path/to/audio --output-dir /path/to/manifests --seed 42`
3. Preprocess each split:
   - `python scripts/phase0/preprocess_audio.py --manifest /path/to/manifests/train.manifest.csv --output-dir /path/to/preprocessed/train --output-manifest /path/to/preprocessed/train.manifest.csv`
4. Validate PESQ/STOI:
   - `python scripts/phase0/validate_pesq_stoi.py --pairs-csv /path/to/pairs.csv --fail-on-invalid`

Phase 1 starter implementation in this repository:

- `scripts/phase1/train_defenders.py`: config-driven Phase 1A training for WavLM, AASIST, RawNet2, and LCNN
- `scripts/phase1/train_attackers.py`: config-driven Phase 1B attacker SFT for Squeezeformer and Mamba against frozen AASIST
- `scripts/phase1/common.py`: shared dataset loading, metrics (EER/AUC/accuracy), checkpointing, and logging helpers
- `scripts/phase1/models.py`: defender and attacker model builders, including WavLM backbone-freeze support
- `scripts/phase1/phase1_exit_checks.py`: validates Phase 1 exit criteria and writes summary JSON
- `scripts/phase1/configs/*.example.json`: example configs for defender and attacker runs

Example usage:

1. Train defenders (Phase 1A):
   - `python scripts/phase1/train_defenders.py --config scripts/phase1/configs/defender_sft.example.json`
2. Train attackers (Phase 1B):
   - `python scripts/phase1/train_attackers.py --config scripts/phase1/configs/attacker_sft.example.json`
3. Run exit checks:
   - `python scripts/phase1/phase1_exit_checks.py --defender-metrics /path/to/defenders/metrics.jsonl --attacker-metrics /path/to/attackers/metrics.jsonl --output-summary /path/to/phase1_exit_summary.json`

Phase 2 starter implementation in this repository:

- `scripts/phase2/stress_test.py`: defender×attacker stress-testing with vulnerability matrix output (EER/AUC/accuracy + attack success + PESQ/STOI)
- `scripts/phase2/configs/stress_test.example.json`: example Phase 2 config

Phase 3 starter implementation in this repository:

- `scripts/phase3/train_rl.py`: 5,000-step zero-sum style RL loop scaffold with defender alternation, attacker schedule switch, curriculum, GRPO-style reward normalization, WGAN-GP-style regularization, and periodic checkpoints/metrics
- `scripts/phase3/configs/rl_train.example.json`: example Phase 3 config

Phase 4 starter implementation in this repository:

- `scripts/phase4/final_evaluation.py`: final evaluation scaffold for FGSM sanity, LCNN transfer, SFT-vs-RL comparison, optional zero-day transfer manifest evaluation, and statistics (DeLong-style AUC p-value + bootstrap EER CI)
- `scripts/phase4/configs/final_eval.example.json`: example Phase 4 config

Pipeline/completion helpers:

- `scripts/phase0/prepare_emofake_dataset.py`: dataset materialization helper (from local source dir or archive URL/path)
- `scripts/pipeline/run_project.py`: phase-by-phase orchestration runner driven by JSON command config
- `scripts/pipeline/configs/full_pipeline.example.json`: full pipeline command template
- `scripts/package_results.py`: completion package exporter and objective summary generator

### Phase 1 — SFT Baselines

#### 1A) Defender SFT

- Train WavLM-head, AASIST, RawNet2, LCNN on clean EmoFake.
- Binary CE loss, AMP FP16, LR `5e-5`, target batch 64 (or gradient accumulation).
- WavLM backbone remains frozen.
- Track EER/AUC/accuracy on dev.

#### 1B) Attacker SFT (frozen AASIST target)

- Train Squeezeformer and Mamba independently against frozen AASIST.
- Objective:
  - maximize defender error (`-L_def`)
  - enforce quality/intelligibility constraints (PESQ >= 2.5, STOI >= 0.75)
  - include WGAN-GP regularization.
- Initial settings: epsilon 0.01, quality penalties lambda1=lambda2=0.5.

Exit criteria:

- Both attackers degrade AASIST EER vs clean baseline
- PESQ/STOI constraints maintained

### Phase 2 — Cross-Architecture Stress Tests

1. **All defenders vs PGD** (`20` steps, `eps=0.01`, `alpha=eps/10`)
2. **All attackers vs frozen WavLM**
3. Build full vulnerability matrix (4 defenders × 6 attackers), logging EER/AUC/accuracy + PESQ/STOI for attacks.

### Phase 3 — 5,000-Step Zero-Sum RL (Core)

Per-step loop:

1. Sample clean real batch.
2. Attacker generates **G=4** adversarial variants per clip.
3. Active defender scores real + variants.
4. Compute rewards:
   - defender reward favors correct fake detection and correct real acceptance
   - attacker reward favors bypass success, penalizes PESQ/STOI violations and speaker-identity drift
5. Update with GRPO:
   - `A_i = (R_i - mean(R_group)) / (std(R_group) + 1e-8)`
   - if reward variance collapses, allow temporary normalization disable/freeze strategy.
6. Apply WGAN-GP penalty (`lambda_gp=10`).

Curriculum:

- **Steps 1–1000**: eps<=0.005, simple emotion pairs, short clips, Squeezeformer
- **Steps 1001–3000**: eps<=0.01, full pairwise emotion transitions, up to 4s
- **Steps 3001–5000**: eps<=0.02, dynamic multi-emotion sequences, up to 8s, switch to Mamba

Defender alternation:

- 1–100 WavLM active, 101–200 AASIST active, repeat every 100 steps.

Monitor every 100 steps:

- dev EER, AUC
- attacker PESQ/STOI
- GRPO reward std
- WGAN-GP stability
- Creativeness Index: `(attack_success_rate * PESQ * STOI) / eps`

### Phase 4 — Final Evaluation

1. RL defender vs FGSM sanity test (expect low EER).
2. RL attacker vs LCNN sanity transfer test (expect high LCNN EER under attack).
3. Main test comparison: SFT defender vs RL defender on locked EmoFake test split.
4. Zero-day transfer: evaluate frozen defenders on ASVspoof 5 CM protocol (no retraining).
5. Statistical validation:
   - DeLong paired test on AUC (target p < 0.05 with correction)
   - 1,000x bootstrap 95% CI for EER (non-overlapping CIs strengthen claim)

---

## 3) Required Metrics and Why They Are Mandatory

- **EER**: primary anti-spoofing claim metric (cannot drop).
- **AUC-ROC**: threshold-independent separability + required for DeLong (cannot drop).
- **Accuracy**: presentation-friendly and imbalance sanity check (optional if constrained).
- **PESQ**: attack quality constraint preventing trivial noisy attacks (cannot drop).
- **STOI**: intelligibility guardrail complementary to PESQ (keep).
- **Creativeness Index (CI)**: tracks adversary creativity under quality budget (cannot drop for project objective).
- **DeLong p-value + bootstrap CI**: required statistical confidence in final improvement claim.

---

## 4) Minimal Reproducibility Rules

- Never regenerate data splits after initial commit.
- Never use test split for checkpoint selection or tuning.
- Keep exact dependency versions pinned across collaborators.
- Save checkpoints and logs at each phase boundary.
