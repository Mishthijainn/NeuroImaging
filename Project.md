# Project Blueprint: A Cross-Attention Deep Learning Framework for Multimodal Neuroimaging Fusion in Vision Defect Detection and Classification

---

## 1. Problem Statement

Automated detection and classification of vision defects (such as Optic Neuritis, Glaucoma, and Cortical Visual Impairments) remain limited in clinical practice due to three primary bottlenecks:

- **Unimodal Isolation:** Clinical diagnosis requires assessing both anatomical structure and electrical transmission latency. Current computational pipelines analyze either 3D structural MRI (sMRI) or 1D electrophysiology (Visual Evoked Potentials - VEP) in isolation. sMRI captures spatial lesions and structural atrophy but lacks millisecond temporal latency. VEP captures conduction delays along the optic nerve but lacks localized spatial resolution.
- **Cross-Modal Heterogeneity:** Merging a high-dimensional 3D volumetric matrix $(1, D, H, W)$ with a 1D temporal waveform $(1, L)$ introduces severe spatial-temporal misalignment. Standard fusion approaches (such as early feature concatenation or late average pooling) treat all feature dimensions equally. This introduces substantial feature noise, overlooks spatial-temporal cross-correlations, and dilutes subtle pathology markers.
- **Lack of Clinical Interpretability:** Conventional deep neural networks operate as black-box decision systems. They do not demonstrate to clinicians _which_ anatomical visual cortex zones (V1, V2, V3) correlate directly with specific millisecond signal delays (such as the $P100$ latency peak).

---

## 2. Proposed Solution

The proposed solution implements an end-to-end **Cross-Attention Fusion Network (CAFN)** designed to bridge 3D structural neuroimaging and 1D electrophysiological latency:

- **Dual-Stream Signal Conditioning:**
- **1D Temporal Stream:** Zero-phase 4th-order Butterworth bandpass filtering ($1.0\text{ Hz} - 100.0\text{ Hz}$) combined with a $50\text{ Hz}$ IIR notch filter to remove electromyographic (EMG) jitter, baseline drift, and AC power-line hum without altering critical $P100$ peak timing.
- **3D Spatial Stream:** Automated NIfTI skull-stripping, RAS orientation alignment, $1.0\text{ mm}$ isotropic voxel resampling, and foreground intensity normalization.

- **Cross-Attention Fusion Module (CAFM):** Rather than concatenating features, spatial 3D volume tokens are projected as Queries ($Q$), while temporal 1D latency tokens are projected as Keys ($K$) and Values ($V$). This enables spatial brain regions to dynamically query electrophysiological latency bins:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Multi-Task Auxiliary Supervision:** Unimodal baseline heads train concurrently with the fused attention head, preventing the network from over-relying on a single dominant modality.
- **Clinical Interpretability Engine:** Generates 2D spatial-temporal cross-attention matrices and 3D visual cortex activation maps, giving clinicians verifiable evidence for classification decisions.

---

## 3. System Architecture & Data Flow

```
========================================================================================
                                PIPELINE FLOW
========================================================================================

   [ 3D sMRI Volume ]                           [ 1D VEP Waveform ]
  (Raw NIfTI: .nii.gz)                          (Raw Electrophysiology)
           │                                               │
           ▼                                               ▼
 [ MONAI Preprocessing ]                        [ Butterworth & Notch ]
   - RAS Reorientation                            - 50 Hz Notch (Mains)
   - 1mm Isotropic Resampling                     - 1-100 Hz Bandpass
   - Intensity Scale [0, 1]                       - Z-Score Normalization
           │                                               │
     (1, 64, 64, 64)                                    (1, 500)
           │                                               │
           ▼                                               ▼
  [ 3D-CNN / ResNet ]                            [ 1D-CNN Latency Net ]
   - 3D Convolutions                              - 1D Temporal Convolutions
   - Adaptive Pool (4x4x4)                        - Adaptive Pool (16)
           │                                               │
    Spatial Tokens (F_s)                           Temporal Tokens (F_t)
       (B, 64, 128)                                   (B, 16, 128)
           │                                               │
           ├──────────────────────────┐                    │
           ▼                          ▼                    ▼
     [ Linear: W_Q ]            [ Linear: W_K ]      [ Linear: W_V ]
           │                                  \      /
       Query (Q)                               Key (K), Value (V)
           │                                      │
           └─────────────────┬────────────────────┘
                             ▼
              [ Multi-Head Cross-Attention ]
               Score = Softmax( Q * K^T / sqrt(d_k) )
               Fused_Tokens = Score * V + Q (Residual)
               LayerNorm & Feed-Forward Block
                             │
                     Fused Tensor (B, 64, 128)
                             │
                             ▼
                    [ Global Mean Pool ]
                             │
                     Embedding (B, 128)
                             │
                             ▼
              [ Softmax Classification Head ]
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
      Healthy Control   Glaucoma     Optic Neuritis / CVI

```

---

## 4. Dataset Repositories & Access Links

| Modality           | Dataset Name                                     | Description / Utility                                                                                     | Access Link                                                                                                           |
| ------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **3D sMRI / fMRI** | **OpenNeuro (ds004496)**                         | High-resolution visual cortex mapping under retinotopic stimulation paradigms.                            | [OpenNeuro ds004496](https://www.google.com/search?q=https://openneuro.org/datasets/ds004496)                         |
| **3D sMRI / fMRI** | **OpenNeuro (ds003787)**                         | Multimodal structural and functional visual pathway stimulation benchmarks.                               | [OpenNeuro ds003787](https://openneuro.org/datasets/ds003787)                                                         |
| **3D sMRI / fMRI** | **Human Connectome Project (HCP) 7T Retinotopy** | Gold-standard 7T structural and retinotopic functional scans of cortical visual areas (V1, V2, V3).       | [HCP Retinotopy Benchmark](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) |
| **3D sMRI**        | **OASIS Brains Database**                        | Large-scale public neuroimaging data for structural normalization and gray matter segmentation baselines. | [OASIS Brains Portal](https://www.oasis-brains.org/)                                                                  |
| **1D VEP / EEG**   | **MNE Sample Electrophysiology**                 | Visual and auditory stimulus responses across occipital sensor lines (e.g., channel EEG 057).             | [MNE Sample Dataset](https://www.google.com/search?q=https://mne.tools/stable/overview/datasets_index.html%23sample)  |
| **1D VEP / ERG**   | **PhysioNet Clinical Electrophysiology**         | Standardized clinical wave files following ISCEV standards for optic pathway conduction analysis.         | [PhysioNet Repositories](https://physionet.org/)                                                                      |

---

## 5. Base Papers & Key References

- **Vaswani, A., et al. (2017).** _Attention Is All You Need._ Advances in Neural Information Processing Systems (NeurIPS).
  _Foundational paper introducing the Query-Key-Value Multi-Head Attention and Transformer architecture._
  Link: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Calhoun, V. D., & Sui, J. (2016).** _Multimodal Fusion of Brain Imaging Data: A Key to Finding the Missing Link(s) in Complex Mental Illness._ Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, 1(3), 230–244.
  _Defines foundational principles for cross-modal neuroimaging integration and information complementarity._
  Link: [DOI: 10.1016/j.bpsc.2015.12.005](https://www.google.com/search?q=https://doi.org/10.1016/j.bpsc.2015.12.005)
- **Zhang, Y., et al. (2024).** _CAMF: Cross-Attention Multimodal Fusion for Neuroimaging-Based Diagnosis of Neurological Disorders._ IEEE Transactions on Medical Imaging.
  _Direct benchmark for cross-attention mechanisms applied to heterogeneous medical imaging modalities._
  Link: [IEEE Xplore / PubMed: 38416629](https://www.google.com/search?q=https://pubmed.ncbi.nlm.nih.gov/38416629/)
- **Hao, R., et al. (2023).** _Cross-Spatial Attention Guided Network for Multimodal Brain Tumor Segmentation and Structural Feature Linking._ Computerized Medical Imaging and Graphics, 108, 102263.
  _Formulates cross-attention modules targeting 3D spatial feature localization._
  Link: [DOI: 10.1016/j.compmedimag.2023.102263](https://www.google.com/search?q=https://doi.org/10.1016/j.compmedimag.2023.102263)
- **Guo, Z., et al. (2025).** _DRIFA-Net: Deformable Cross-Attention and Iterative Feature Alignment for Multimodal 3D Medical Volume Fusion._ IEEE Winter Conference on Applications of Computer Vision (WACV).
  _State-of-the-art methodology for 3D alignment and cross-attention feature aggregation._
  Link: [WACV 2025 Open Access / arXiv:2411.08324](https://arxiv.org/abs/2411.08324)
- **Odom, J. V., et al. (2016).** _ISCEV Standard for Clinical Visual Evoked Potentials (2016 Edition)._ Documenta Ophthalmologica, 133(1), 1–9.
  _Clinical protocol defining standard VEP waveform biomarkers, calibration, and P100 peak latency metrics._
  Link: [DOI: 10.1007/s10633-016-9553-y](https://www.google.com/search?q=https://doi.org/10.1007/s10633-016-9553-y)

---

## 6. Implementation Summary & Deliverables

```
├── data/
│   ├── raw/
│   │   ├── mri/                     # Raw 3D NIfTI volumes (.nii.gz)
│   │   └── vep/                     # Raw 1D electrophysiological CSVs (.csv)
│   └── processed/                   # Resampled, skull-stripped, and filtered arrays
├── src/
│   ├── preprocessing/
│   │   ├── signal_cleaner.py        # Butterworth & notch filtering
│   │   └── mri_transforms.py        # MONAI spatial standardizer
│   ├── models/
│   │   ├── spatial_encoder.py       # 3D-CNN / ResNet backbone
│   │   ├── temporal_encoder.py      # 1D-CNN latency backbone
│   │   └── cross_attention.py       # Multi-Head CAFM layer
│   ├── train.py                     # Multi-task training loop
│   └── explainability/
│       ├── grad_cam_3d.py           # 3D spatial activation maps
│       └── plot_attention.py        # 2D cross-attention correlation heatmaps
├── checkpoints/                     # Serialized PyTorch models (.pt)
└── requirements.txt                 # torch, monai, nibabel, nilearn, mne, scipy

```
