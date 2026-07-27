# Spectral-LDM

A class-conditional **latent diffusion model for 1D FTIR spectra**, and the leak-free evaluation
harness used to test whether generative augmentation helps downstream cancer classification.

Reference implementation for *A Latent Diffusion Framework for Generative Augmentation of FTIR
Spectroscopy*.

---

## Reproduce everything in one run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NazimBL/SpectralDiffusionLab/blob/master/Full_Evaluation.ipynb)

**[`Full_Evaluation.ipynb`](Full_Evaluation.ipynb)** runs the entire pipeline from a clean
environment and produces every table and figure in the paper. Open it in Colab with the badge above,
select a GPU runtime, and run the cells in order.

```
Part 1  Environment and data ....... clone, patch, patient-level Kennard–Stone split
Part 2  Generative models .......... autoencoder → latent cache → DDPM → cWGAN-GP baseline
Part 3  Evaluation harness ......... leak-free training, bootstrap CIs, DeLong, thresholds
Part 4  Experiment I ............... five balancing strategies compared
Part 5  Experiment II .............. augmentation ratio swept 0.4×–2.0×
Part 6  Collect outputs ............ single archive of checkpoints and results
```

Runtime is roughly **60–110 minutes** on a T4, dominated by Part 2. With checkpoints already
present, Parts 4 and 5 take about 15 and 25 minutes on their own.

The notebook is self-contained: it writes the evaluation harness to disk, applies the source patches
listed below, and needs no manual file transfers.

---

## What produces what

| Paper artefact | Produced by |
|---|---|
| **Table 1** — balancing strategies, screening operating point | `threshold_analysis.py` via Part 4.2 |
| **Table 2** — DeLong tests, bootstrap intervals | `eval_utils.py` via Part 4.1 |
| **Table 3** — Youden operating point | `threshold_analysis.py` via Part 4.2 |
| **Table 4** — fixed 0.5 cutoff, five seeds | `eval_utils.py` via Part 4.1 |
| **Table 5 / Figure 3** — augmentation-ratio sweep | Part 5 |
| **Figure 1** — framework overview | schematic, not generated |
| **Figure 2** — generated vs. real spectra | `QC.py` |
| **Figure 4** — U-Net architecture | schematic, not generated |

---

## Repository map

### Active pipeline

| Path | Role | Paper section |
|---|---|---|
| `data/data_preparation.py` | Kennard–Stone split at patient level, excludes atypical hyperplasia, averages each patient's five replicate acquisitions | §3.1 |
| `train_ae.py` | 1D convolutional autoencoder defining the latent space | §3.2 |
| `Latent_ddpm_z.py` | Encodes training spectra into the latent cache | §3.2 |
| `train_ddpm_latent.py` | Class-conditional latent DDPM with classifier-free guidance | §3.3 |
| `cGAN.py` | Adversarial baseline — a conditional **Wasserstein** GAN with gradient penalty, stronger than the `cGAN` name suggests | §4 |
| `experiment_balancing.py` | Model definitions and sampling routines for the balancing comparison | §4.2 |
| `QC.py` | Quality-control comparison of generated against real spectra | §4.1 |
| `eval_utils.py` | Leak-free XGBoost training, stratified bootstrap CIs, DeLong test, multi-seed harness | §3.6 |
| `threshold_analysis.py` | Validation-selected operating points (Youden's *J*, sensitivity floor) | §3.6 |
| `Full_Evaluation.ipynb` | End-to-end reproduction | all |

### Evaluation harness

The two modules that carry the paper's protocol:

**`eval_utils.py`**
- `train_xgb_no_leak` — early stopping on a validation split drawn from the **training partition
  only**, so the test set never influences the number of boosting rounds
- `bootstrap_ci` — stratified percentile confidence intervals
- `delong_test` — DeLong's test for two correlated ROC curves
- `evaluate_strategy` — multi-seed harness taking a `build_train_fn(seed)` callable, so stochastic
  augmentation is regenerated per seed and the reported standard deviations capture **generator**
  variance rather than classifier initialisation alone

**`threshold_analysis.py`**
- `evaluate_strategy_thresholded` — selects the decision threshold on validation data under a named
  rule, freezes it, then applies it once to the test set

### Exploratory and superseded code

These directories are kept for provenance and are **not** part of the reported results:

| Path | Contents |
|---|---|
| `latent space diffusion v0/` | Earlier latent-space experiments, including a VQ-VAE variant and the composite-loss DDPM described below |
| `legacy/` | Raw-domain (non-latent) diffusion, PLS-DA and PET baselines, hyperparameter sweeps |
| `vanilla-compare/` | Standalone balancing and LDM-vs-GAN comparison scripts predating the current harness |
| `Strategic_Augmentation_Comparison/`, `Balancing_Comparison_Final_All/` | Result CSVs from earlier runs |

**`latent space diffusion v0/triplet_ddpm_latent.py`** is worth singling out. It trains the same
latent DDPM under a composite objective rather than plain noise prediction:

```
L = λ_mse · MSE(ε, ε̂)                    (λ = 1.0)
  + λ_peak · peak-weighted MSE(decoded)   (λ = 0.1)
  + λ_triplet · batch-all triplet loss    (λ = 0.1, margin 0.2)
```

The peak term applies Gaussian weights centred on assigned biochemical bands
(1716, 1446, 1377, 1234, 1045, 900 cm⁻¹), so reconstruction error at diagnostically meaningful
wavenumbers is penalised more heavily than error elsewhere. The triplet term pushes class
embeddings apart.

It writes to `ldm_out_triplet/` and is **not** the model evaluated in the paper, which uses
`train_ddpm_latent.py` and `ldm_out/`.

---

## Source patches

`Full_Evaluation.ipynb` applies three fixes in place at run time. They are mechanical and change no
modelling decision:

1. **Path normalisation.** `Latent_ddpm_z.py` reads `../MyDataset/` while every other script uses
   `MyDataset/`.
2. **Removal of test-set early stopping.** `experiment_balancing.py` passes the test set as
   XGBoost's `eval_set`, which makes the number of boosting rounds a hyperparameter fitted to the
   test data. All reported evaluation instead goes through `eval_utils.py`.
3. **Wavenumber column names.** `data_preparation.py` emits generic `X.*` names; the training
   scripts expect numeric wavenumbers.

Results in `Balancing_Comparison_Final_All/` and `Strategic_Augmentation_Comparison/` predate
fix (2) and are retained only for provenance. **Do not cite numbers from those CSVs.**

---

## Dataset

ATR-FTIR blood plasma spectra from Paraskevaidi et al. (2020),
[*Cancers* 12(5):1256](https://doi.org/10.3390/cancers12051256).

After excluding atypical hyperplasia: **584 patients**, split at patient level into 409 training
(168 Healthy, 241 Cancer) and 175 held-out test (74 Healthy, 101 Cancer). Each patient contributes
five replicate acquisitions, averaged to one patient-level spectrum before any modelling — so the
unit of analysis is the patient throughout, and no replicate of a test patient is seen during
training.

Preprocessing: Savitzky–Golay smoothing (window 5, order 2), second-derivative transform,
L2 normalisation. Parameters fixed *a priori* and applied identically to both partitions.

---

## Local installation

```bash
git clone https://github.com/NazimBL/SpectralDiffusionLab.git
cd SpectralDiffusionLab
pip install torch xgboost imbalanced-learn scikit-learn scipy pandas matplotlib
```

Then follow the same order as the notebook:

```bash
python data/data_preparation.py     # build the split
python train_ae.py                  # autoencoder
python Latent_ddpm_z.py             # cache latents
python train_ddpm_latent.py         # latent diffusion model
python cGAN.py                      # adversarial baseline
```

A GPU is strongly recommended; the diffusion model is impractical to train on CPU.

---

## Reproducibility notes

`torch.manual_seed` fixes the RNG but not cuDNN kernel selection, so GPU runs of the stochastic
generators can differ slightly between sessions while deterministic strategies (Original, SMOTE)
reproduce exactly. For bit-reproducible runs, set `torch.use_deterministic_algorithms(True)` and
`torch.backends.cudnn.deterministic = True` before training, at some cost in speed.

Every reported result uses five seeds with augmentation regenerated per seed.

---

## Citation

```bibtex
@misc{spectralldm,
  author = {Belabbacci, Nazim A. and Kovalev, Anton and Anaadumba, Raphael
            and Alam, Mohammad Arif Ul},
  title  = {A Latent Diffusion Framework for Generative Augmentation of
            FTIR Spectroscopy},
  year   = {2027},
  note   = {University of Massachusetts Lowell}
}
```

If you use the dataset, cite Paraskevaidi et al. (2020) as its source.
