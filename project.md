# GlioGrade — Full Project Context

> This file is the single source of truth for anyone (human or LLM) picking up this project.
> It covers medical background, what has been built, what is broken, every technical decision made,
> and the full plan for what comes next. Read this before touching any code.

---

## Table of Contents

1. [What is GlioGrade?](#1-what-is-gliograde)
2. [Medical Background](#2-medical-background)
3. [Datasets](#3-datasets)
4. [What Was Built (Current State)](#4-what-was-built-current-state)
5. [Why the Models Barely Beat Random](#5-why-the-models-barely-beat-random)
6. [The Full Plan Going Forward](#6-the-full-plan-going-forward)
7. [Repo Structure](#7-repo-structure)
8. [Workflow: Local vs Google Colab](#8-workflow-local-vs-google-colab)
9. [Preprocessing Pipeline](#9-preprocessing-pipeline)
10. [Model Architecture](#10-model-architecture)
11. [Training Strategy](#11-training-strategy)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Notebook Pipeline](#13-notebook-pipeline)
14. [Website Overhaul](#14-website-overhaul)
15. [Open Questions and Decisions Still Needed](#15-open-questions-and-decisions-still-needed)

---

## 1. What is GlioGrade?

GlioGrade is a locally hosted web tool for AI-assisted glioma diagnosis from 3D MRI scans.
A clinician uploads a NIfTI scan (`.nii` or `.nii.gz`), and the tool returns:

- **Tumor type** — one of 4 WHO 2021 classes
- **Tumor grade** — one of 3 grades (2, 3, 4)
- **Confidence scores** for each prediction
- **Interactive MRI viewer** — axial, coronal, and sagittal slice browser

The tool is designed to run locally within a medical institution (no cloud uploads of patient data).

**Key novelty:** First tool to combine typing *and* grading using 3D CNNs trained on the **WHO 2021** classification. Most prior work uses 2D CNNs and/or the outdated 2016 WHO classification.

**Initial results (Syslab, May 2025):** 84.57% typing accuracy, 83.84% grading accuracy — but significant improvements are planned. See Section 5 for the full analysis and fix plan.

---

## 2. Medical Background

### Glioma types (WHO 2021 classification — 4 classes)

| Label (exact string in metadata) | Short name | Notes |
|---|---|---|
| `Glioblastoma, IDH-wildtype` | GBM | Most common, most aggressive, ~80% of dataset |
| `Astrocytoma, IDH-wildtype` | Astro WT | Rare in dataset |
| `Oligodendroglioma, IDH-mutant, 1p/19q-codeleted` | Oligo | ~16% of dataset |
| `Astrocytoma, IDH-mutant` | Astro MT | ~20% of dataset |

**Important:** In 2021 WHO reclassified gliomas. IDH-mutant astrocytomas are no longer called glioblastomas. Many prior ML models are trained on the 2016 scheme and are outdated.

### Glioma grades (3 classes)

| Grade | Severity | Count in UCSF |
|---|---|---|
| 2 | Low — slow growing | 55 (11%) |
| 3 | Mid — moderately aggressive | 42 (9%) |
| 4 | High — most aggressive | 403 (**80%**) |

### MRI modalities in UCSF-PDGM (per patient)

Each patient folder contains: `T1`, `T1_bias`, `T2`, `T2_bias`, `T2_FLAIR`, `T1c`, `T1c_bias`, `SWI`, `FLAIR`, `DWI`, `ASL`, `ADC`, segmentation files.

The model was trained on **T2_bias** scans (bias-field corrected T2).
The UCSF dataset is already skull-stripped and normalized — no manual preprocessing needed for training.

---

## 3. Datasets

### UCSF-PDGM (primary — training)

- **Source:** The Cancer Imaging Archive (TCIA) — https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- **Patients:** 495 (some IDs are follow-up scans of the same patient)
- **Classification:** WHO 2021
- **Size:** ~156 GB full dataset; ~1 GB for T2_bias scans only
- **Preprocessing (done by UCSF, confirmed on TCIA page):**
  - Skull-stripped ✅
  - Bias-field corrected ✅ — the `_bias` suffix in the filename IS the bias correction label
  - Co-registered to T2/FLAIR space at **1mm isotropic resolution** using ANTs ✅
  - Converted to NIfTI format ✅
  - **We do not need to run skull stripping or N4 bias correction ourselves.**
- **Metadata file:** `UCSF-PDGM-metadata_v2.csv`
  - Key column: `Final pathologic diagnosis (WHO 2021)`
  - ID column: `ID` (format: `UCSF-PDGM-XXXX`)

### Confirmed Drive folder layout (audited manually, May 2026)

| Drive path | Contents | Use? |
|---|---|---|
| `MyDrive/UCSF_data/T2biasCollected/` | Raw UCSF-PDGM T2_bias NIfTIs, files named `UCSF-PDGM-XXXX_T2_bias.nii.gz`. No notebook ever wrote to this folder — it is original data. | **✅ Use this for training** |
| `MyDrive/UCSF_data/T2Typing/T2Training/` | T2_bias scans pre-organized into 4 WHO class subfolders with a train/val split | Reference only — derived from above |
| `MyDrive/data/UCSF_T2Flair/T2FlairCollected/` | UCSF-PDGM FLAIR scans (different modality, same patients). Files named `UCSF-PDGM-XXXX_FLAIR.nii.gz` | Not used for now |
| `MyDrive/data/EGD_data_T2/` | Erasmus (EGD) raw T2 scans — WHO 2016 labels | ❌ Skip |
| `MyDrive/data/EGD_data_T2/stripped/` | EGD skull-stripped output | ❌ Skip |
| `MyDrive/data/EGD_data_T2/biased+stripped/` | EGD N4-corrected output (also resized to 64³ in some versions) | ❌ Skip |

**Confirmed metadata path:** `MyDrive/UCSF_data/UCSF-PDGM-metadata_v2.csv`

### Erasmus Glioma Database (out of scope)

- **Patients:** 774
- **Classification:** WHO 2016 (incompatible with our WHO 2021 labels)
- **Decision:** Skip entirely. Focus on UCSF only.

---

## 4. What Was Built (Current State)

### Flask web app (`GlioGrade/app.py`)

- User uploads a single `.nii`/`.nii.gz` file
- App loads two Keras models at startup: `models/type_model.keras`, `models/grade_model.keras`
- **BUG:** Inference is 2D — takes the middle axial slice, resizes to 224×224, runs through model
- The actual 3D models were never wired into `app.py` before the demo deadline
- Generates axial/coronal/sagittal slice PNGs served to the frontend
- Results page has an interactive slice browser (orientation dropdown + slice slider)

### Templates

| File | Purpose |
|---|---|
| `base.html` | Shared layout, header, animated mesh gradient background |
| `home.html` | Landing page with hero + 3 feature cards |
| `upload.html` | File picker (custom styled, hidden `<input>`) |
| `results.html` | Shows type/grade/confidence + interactive MRI viewer |

### Known bugs / inconsistencies in the current app

- Inference is 2D (single middle slice), not 3D — does not match the trained models
- `upload.html` says "T2 MRI file" but `app.py` saves it as `T1.nii.gz`
- `results.html` labels the viewer "T2 Scan" but that's meaningless since it's whatever the user uploaded
- `home.html` says "GliomaDx" in one place (old name)
- `style.css` has a `hero-bg` referencing a dead placeholder image URL (`via.placeholder.com`)
- No loading indicator while the server generates all slice PNGs (takes several seconds)
- `models/` directory is empty — no models are committed to the repo

### Previous model architecture (3D CNN)

Described in the presentation (PyTorch-style):

```
Input: (B, 1, 64, 64, 64)
Conv3D(1→32) + BatchNorm3D + ReLU
MaxPool3D                          → (B, 32, 32, 32, 32)
Conv3D(32→64) + BatchNorm3D + ReLU
MaxPool3D                          → (B, 64, 16, 16, 16)
Conv3D(64→128) + BatchNorm3D + ReLU
MaxPool3D                          → (B, 128, 8, 8, 8)
Conv3D(128→256) + BatchNorm3D + ReLU
MaxPool3D                          → (B, 256, 4, 4, 4)
Flatten                            → (B, 16384)
FC(16384→512) + ReLU + Dropout
FC(512→num_classes)
```

This was trained in TensorFlow/Keras on Colab. The training notebooks exist somewhere on Google Drive.

---

## 5. Why the Models Barely Beat Random

This section is the most important one. Here is the full root-cause analysis of why the models underperformed and what we are doing to fix it:

### Root cause 1: Class imbalance (almost certainly the primary issue)

Grade 4 makes up **80%** of the dataset. A model that always predicts "Grade 4" achieves **80% accuracy** without learning anything. For typing, Glioblastoma IDH-wildtype is similarly dominant.

If plain accuracy was used as the training metric, the model's gradient signal was dominated by the majority class. It learned to predict Grade 4 / GBM for everything, achieving a deceptively high accuracy number.

**Fix:** Inverse-frequency class-weighted cross-entropy loss + switch primary metric to balanced accuracy / macro F1 / Cohen's Kappa.

### Root cause 2: Wrong evaluation metric

Plain accuracy is misleading for imbalanced classification. It masks the model predicting only the majority class.

**Fix:** Use balanced accuracy, macro F1, Cohen's Kappa. Never report plain accuracy as the headline metric on this dataset.

### Root cause 3: No stratified split

If the train/val/test split was done randomly without stratification, the minority classes (Grade 2: 55 samples, Grade 3: 42 samples) may have been almost entirely in training or almost entirely absent from validation, causing misleading validation numbers.

**Fix:** Always use `sklearn.model_selection.StratifiedShuffleSplit` for splits.

### Root cause 4: Training from scratch on ~495 samples

The 3D VGG-style architecture has millions of parameters. 495 samples is not enough to train it from scratch. The model will overfit the training set with high training accuracy but poor validation accuracy.

**Fix:** Use a pretrained 3D model (MedicalNet — see Section 10) as the backbone. Fine-tune only the final layers first, then unfreeze gradually.

### Root cause 5: Suboptimal normalization

Simple min-max normalization on the full volume is sensitive to outlier voxels (scanner noise, bright spots). If the intensity range is dominated by one extreme value, most of the volume gets compressed to near-zero.

**Fix:** Z-score normalization computed on non-zero (brain) voxels only, then clip to [-3, 3] standard deviations, then rescale to [0, 1].

### Root cause 6: Input volume may not be centered on tumor

Resizing the full 64³ brain volume means most of the volume is healthy brain tissue. The tumor may occupy only a small fraction of voxels, making it hard for the CNN to find the relevant signal.

**Partial fix (without segmentation):** Crop to 96³ or 128³ centered on the scan instead of the full volume, or use a larger input size.

### What we are NOT doing (segmentation approach, ruled out for now)

The poster describes multiplying T2 FLAIR × segmentation mask to isolate the tumor and expand the value range. This worked well but requires 4 MRI modalities (T1c, T1, T2, FLAIR) and running SegResNet first. We do not currently have all 4 modality files available for enough patients, so this is deferred.

---

## 6. The Full Plan Going Forward

### Phase 1: Data audit — COMPLETE (done manually, May 2026)

The data audit was done by reviewing all legacy training notebooks directly rather than running automated tooling.

**Findings:**
- Training data confirmed at `MyDrive/UCSF_data/T2biasCollected/` — original UCSF-PDGM T2_bias NIfTIs, never modified by any notebook
- UCSF-PDGM data is already skull-stripped, bias-corrected, and 1mm isotropic — confirmed on TCIA and in source code (N4 bias correction was explicitly skipped for UCSF in notebooks)
- Modality to use: **T2_bias** (consistent with best training notebooks)
- The old preprocessing pipeline had two critical data bugs: (1) min-max normalization over full volume including zero-background voxels, and (2) using `tf.image.resize` on 2 dims + `np.resize` on the third — this corrupts 3D structure entirely
- EGD (Erasmus) data is irrelevant — WHO 2016 labels, kept separately in `EGD_data_T2/`
- See `training/legacy/ANALYSIS_REPORT.md` for the full audit findings

**What this means for preprocessing:**
- ❌ Skip: skull stripping (done by UCSF)
- ❌ Skip: bias correction / N4 (done by UCSF — `_bias` suffix confirms it)
- ✅ Run: z-score normalization on brain voxels only
- ✅ Run: proper 3D resize to 96³ (using `scipy.ndimage.zoom`, not `tf.image.resize`)
- ✅ Run: save as `.npy` + build manifest CSV

### Phase 2: Fix training pipeline
- Stratified splits
- Z-score normalization
- Weighted cross-entropy loss
- Balanced accuracy / F1 / Kappa as primary metrics
- Data augmentation (random flip, rotation, intensity jitter)

### Phase 3: Validate on binary task first
Before 4-class typing and 3-class grading, prove the pipeline works on binary grading:
- **Low-grade** (Grade 2 + 3, n=97) vs **High-grade** (Grade 4, n=403)
- Target: balanced accuracy > 65% (well above the ~50% random baseline for binary)
- If this doesn't work, something fundamental is still broken

### Phase 4: Switch to MedicalNet pretrained backbone
Replace the VGG-style 3D CNN with a pretrained 3D ResNet-10 from MedicalNet (Tencent).
- Pretrained on large medical imaging datasets (not just ImageNet)
- 10× faster convergence than training from scratch
- Much better generalization on small datasets like ours
- MIT license, available on HuggingFace and GitHub

### Phase 5: Full 3-class and 4-class training
Once binary grading is validated, train:
1. Grade model: 3 classes (Grade 2, 3, 4) with weighted loss
2. Type model: 4 classes (GBM, Astro WT, Oligo, Astro MT) with weighted loss

### Phase 6: Wire up 3D inference in app.py
Replace the 2D middle-slice inference with proper 3D volume loading and inference.
Update the upload flow to accept the correct scan type (T2_bias).

### Phase 7: Website overhaul
Visual and functional improvements to the Flask app (see Section 14).

---

## 7. Repo Structure

```
GlioGrade/
├── GlioGrade/                    ← Flask web app
│   ├── app.py
│   ├── static/
│   │   ├── css/style.css
│   │   └── logo.png
│   └── templates/
│       ├── base.html
│       ├── home.html
│       ├── upload.html
│       └── results.html
│
├── training/                     ← All ML code
│   ├── notebooks/
│   │   ├── 01_preprocessing.ipynb   ← z-score norm, resize, save .npy + manifest
│   │   ├── 02_eda.ipynb             ← class distribution, volume stats, vis
│   │   ├── 03_train_binary.ipynb    ← binary LGG vs HGG (pipeline validation)
│   │   ├── 04_train_grade.ipynb     ← 3-class grade model
│   │   ├── 05_train_type.ipynb      ← 4-class type model
│   │   └── 06_evaluation.ipynb      ← confusion matrix, F1, Kappa, AUC
│   └── src/                         ← importable Python modules (used by notebooks)
│       ├── dataset.py               ← PyTorch Dataset class for NIfTI volumes
│       ├── models.py                ← MedicalNet wrapper + classification head
│       ├── transforms.py            ← z-score norm, augmentation
│       └── metrics.py               ← balanced accuracy, Kappa, per-class F1
│
├── models/                       ← trained .keras / .pt weight files
│   └── .gitkeep                  ← models are gitignored, never committed
│
├── CONTEXT.md                    ← this file
├── README.md
├── requirements.txt
└── .gitignore
```

**Note:** `models/` is in `.gitignore`. Model weights live on Google Drive and are downloaded locally after training. They are never committed to the repo.

---

## 8. Workflow: Local vs Google Colab

### Rule: Repo = source of truth for code. Google Drive = data + model weights.

```
GitHub repo                Google Drive                  Local machine
───────────                ─────────────                 ─────────────
training/notebooks/ ──→   Mount in Colab       ──→      Download .pt/.keras
training/src/       ──→   Run training                   after training
GlioGrade/                                               models/  (gitignored)
                          UCSF-PDGM data                 Run Flask app locally
                          Preprocessed versions
                          Model checkpoints
                          Old training .ipynb files
```

### Opening notebooks on Colab

`File > Open notebook > GitHub` → paste repo URL → select notebook.
No manual upload needed.

### Standard first two cells of every training notebook

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR  = '/content/drive/MyDrive/UCSF_data/T2biasCollected'   # confirmed
META_CSV  = '/content/drive/MyDrive/UCSF_data/UCSF-PDGM-metadata_v2.csv'
MODEL_OUT = '/content/drive/MyDrive/GlioGrade/models'

# Cell 2 — Pull latest src/ from repo
!git clone https://github.com/YOUR_USERNAME/GlioGrade.git /content/GlioGrade
import sys
sys.path.insert(0, '/content/GlioGrade/training/src')
```

### After training

1. Download the `.pt` or `.keras` file from Google Drive to local `models/`
2. Run `python GlioGrade/app.py` locally to test inference
3. If you edited notebook code in Colab: `File > Save a copy to GitHub` to push back

---

## 9. Preprocessing Pipeline

### What the UCSF-PDGM data already has — confirmed, do not repeat these

- ✅ Skull-stripped (skull voxels = 0, zero fraction > 30% of volume)
- ✅ Bias-field corrected — the `_bias` in `T2_bias.nii.gz` is the correction flag. Do NOT run N4 again.
- ✅ Co-registered to T2/FLAIR space at 1mm isotropic resolution (ANTs)
- ✅ Converted to NIfTI

**The old notebooks incorrectly applied bias correction and skull stripping to the UCSF data in some branches — those runs were either errored out or run on EGD data only. T2biasCollected is untouched original data.**

### What we do in the preprocessing notebook (01_preprocessing.ipynb)

1. **Load T2_bias NIfTI** for each patient via `nibabel`
2. **Z-score normalization** on non-zero (brain) voxels only:
   ```python
   brain_mask = volume > 0
   mean = volume[brain_mask].mean()
   std  = volume[brain_mask].std()
   volume = (volume - mean) / (std + 1e-8)
   volume = np.clip(volume, -3, 3)
   volume = (volume + 3) / 6.0   # rescale to [0, 1]
   ```
3. **Resize to (96, 96, 96)** using `scipy.ndimage.zoom` or `skimage.transform.resize` with anti-aliasing
4. **Save as `.npy`** per patient for fast loading during training
5. **Build a manifest CSV** mapping patient ID → `.npy` path → label (type + grade)

### What we are NOT doing (deferred)

- Segmentation preprocessing (requires 4 modalities + SegResNet)
- Skull stripping (UCSF is already stripped)
- Erasmus dataset preprocessing (Erasmus is out of scope for now)

### Data augmentation (applied during training, not saved to disk)

- Random left-right flip (p=0.5) — anatomically valid for brain
- Random rotation ±10° around all three axes
- Random intensity scale ×[0.9, 1.1] + shift ±0.05
- Do NOT use elastic deformation without validating it preserves tumor structure

---

## 10. Model Architecture

### Decision: Use MedicalNet pretrained 3D ResNet-10 as backbone

**Why not the original VGG-style 3D CNN from scratch:**
- 495 samples is too few to train a multi-million parameter network from scratch
- VGG-style (no skip connections) is prone to vanishing gradients in 3D
- No pretraining means convergence takes many more epochs and generalizes poorly

**MedicalNet (Tencent):**
- 3D ResNet variants pretrained on large medical imaging datasets (not Kinetics/ImageNet)
- Documented 3–20% accuracy improvement and 10× faster convergence on small datasets
- ResNet-10 is the right size for our dataset (lightest variant, fewest parameters)
- MIT license
- GitHub: https://github.com/Tencent/MedicalNet
- HuggingFace: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10

**Architecture plan:**

```python
# Backbone: MedicalNet ResNet-10 pretrained encoder
backbone = MedicalNet_ResNet10(pretrained=True)

# Freeze backbone initially, train only head
for param in backbone.parameters():
    param.requires_grad = False

# Classification head (added on top)
head = nn.Sequential(
    nn.AdaptiveAvgPool3d(1),
    nn.Flatten(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)

# After head converges (~10 epochs), unfreeze backbone and fine-tune with low LR
```

**Two separate models** (same architecture, different num_classes):
- Type model: `num_classes = 4`
- Grade model: `num_classes = 3`

**Input shape:** `(B, 1, 96, 96, 96)` — single channel (grayscale MRI), 96³ volume

---

## 11. Training Strategy

### Class weighting (critical — addresses root cause 1)

```python
from sklearn.utils.class_weight import compute_class_weight
import torch

weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Stratified split

```python
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
# Do this twice: first to carve off test, then again to split remainder into train/val
# Final split: 70% train / 15% val / 15% test
```

### Optimizer and schedule

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

- Phase 1 (epochs 1–10): Backbone frozen, LR = 1e-3 for head only
- Phase 2 (epochs 11+): Unfreeze backbone, LR = 1e-4 for all parameters

### Other training details

- Batch size: 4–8 (3D volumes are memory-intensive)
- Mixed precision training (`torch.cuda.amp`) — cuts memory roughly in half
- Early stopping on validation balanced accuracy with patience = 15
- Save best checkpoint (not last epoch)

---

## 12. Evaluation Metrics

**Never use plain accuracy as the headline metric on this dataset.** Grade 4 is 80% of the data — a model predicting only Grade 4 would score 80% accuracy while being clinically useless.

### Primary metrics (report all of these)

| Metric | Why |
|---|---|
| **Balanced accuracy** | Mean per-class recall — not skewed by majority class |
| **Macro F1** | F1 averaged equally across all classes |
| **Cohen's Kappa** | Agreement above chance: 0 = no better than random, >0.6 = meaningful |
| **Confusion matrix** | Always visualize — shows exactly which classes are confused |
| **Per-class precision/recall/F1** | Shows which classes the model actually learns vs ignores |

### Secondary metrics

| Metric | Why |
|---|---|
| AUC-ROC (one-vs-rest) | Good for imbalanced; measures ranking quality per class |
| Training vs validation loss curves | Diagnose overfitting |

### Validation checkpoints for binary task (notebook 03)

Before moving to multi-class training, the binary LGG vs HGG model must clear:
- Balanced accuracy > 0.65 on validation set
- Cohen's Kappa > 0.25
- Confusion matrix shows both classes being predicted (not just one)

If these are not met, stop and diagnose before training the full 3-class/4-class models.

---

## 13. Notebook Pipeline

Each notebook is self-contained and Colab-ready. All notebooks start with Drive mount + repo clone cells.

| Notebook | Purpose | Inputs | Outputs |
|---|---|---|---|
| `01_preprocessing.ipynb` | Z-score normalize + resize volumes | Raw T2_bias NIfTIs from `T2biasCollected/` + metadata CSV | `.npy` files + manifest CSV |
| `02_eda.ipynb` | Explore dataset | Manifest CSV + `.npy` files | Class distribution plots, volume stats, sample visualizations |
| `03_train_binary.ipynb` | Binary LGG vs HGG (pipeline validator) | Preprocessed volumes | Binary model checkpoint, metrics |
| `04_train_grade.ipynb` | 3-class grade model | Preprocessed volumes | Grade model `.pt` |
| `05_train_type.ipynb` | 4-class type model | Preprocessed volumes | Type model `.pt` |
| `06_evaluation.ipynb` | Full evaluation | Model checkpoints + test set | Confusion matrices, F1, Kappa, AUC, per-class breakdown |

**Run order:** 01 → 02 → 03 (validate) → 04 → 05 → 06

**Note:** `00_data_audit.ipynb` and `00b_compare_audits.ipynb` were deleted — the data audit was completed manually by reviewing legacy notebooks. Findings are in `training/legacy/ANALYSIS_REPORT.md`.

---

## 14. Website Overhaul

### Functional fixes (required to work with real 3D models)

- `app.py`: Replace 2D middle-slice inference with full 3D volume load → resize → normalize → model forward pass
- `upload.html`: Update copy to say "T2 MRI file (.nii or .nii.gz)" and clarify what scan type is expected
- `results.html`: Fix label "T2 Scan" on viewer
- `home.html`: Fix stale "GliomaDx" reference
- Add a **loading/processing screen** between upload and results (slice generation takes several seconds — currently the browser just hangs)
- Add proper error page for bad file types / corrupt NIfTI

### Visual fixes

- `style.css`: Remove dead `via.placeholder.com` background image URL from `.hero-bg`
- Add confidence bar visualizations on results page (instead of plain text percentages)
- General polish pass

### Not doing (scope limit)

- Multi-file upload (would require segmentation pipeline — deferred)
- User accounts / scan history
- Any cloud deployment (stays locally hosted by design)

---

## 15. Resolved Questions and Remaining Decisions

### Resolved (May 2026 audit)

- **Which folder to use?** `MyDrive/UCSF_data/T2biasCollected/` — confirmed original, unmodified data.
- **Which scan type?** T2_bias — consistent across the most complete training notebooks, already bias-corrected.
- **Do we need skull stripping?** No — UCSF-PDGM is pre-stripped. Confirmed on TCIA and in source notebooks.
- **Do we need bias correction?** No — `_bias` suffix confirms it is already done. Old notebooks explicitly skipped N4 for UCSF.
- **What did the old pipeline do wrong?** Two critical bugs: (1) min-max normalization over the full volume including zero background, (2) `tf.image.resize` on 2D slices + `np.resize` for the third axis — this completely corrupts 3D spatial structure. Plus 64³ was too small (4× downsampling destroys tumor detail).
- **What preprocessing is still needed?** Only: z-score normalization on brain voxels + proper 3D resize to 96³ + save as `.npy` + manifest CSV.
- **Is the EGD data usable?** No — WHO 2016 labels, incompatible. Skip entirely.

### Still open

- **Exact patient count in T2biasCollected:** Expected ~495 but not verified on Drive. Run a quick `len(os.listdir(...))` check in Colab before preprocessing.
- **Any patients with missing/misc labels in metadata?** The metadata has some labels outside the 4 WHO 2021 classes — these must be dropped. The exact count is unknown until preprocessing runs.
- **Input size 96³ vs 128³?** 96³ is the plan. If Colab A100/V100 has enough VRAM, 128³ is worth trying (better spatial resolution for the CNN). Decide at preprocessing time based on the actual volume dimensions from T2biasCollected.
- **Single model vs two separate models?** Current plan: two separate models (grade + type). A shared backbone with two heads is an option worth considering if GPU memory is tight.

---

*Last updated: May 2026 — based on full review of poster, presentation slides, codebase, and research into class imbalance + transfer learning best practices for 3D MRI classification.*
