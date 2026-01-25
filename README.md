# Impulse 2026 – Self-Supervised Audio Retrieval

This repository contains our submission for **Impulse 2026**, a national-level Signal Processing and Machine Learning hackathon focused on **Audio Signal Processing**.

## Problem Overview
The objective is to learn meaningful audio representations **without labels** and use them for robust audio retrieval, similar to a Shazam-style identification system.

## Methodology
We follow a structured, end-to-end design aligned with the problem statement:

- **Phase 1 – Audio Representation**
  - Raw audio preprocessing
  - MFCC-based time–frequency feature extraction
  - Data augmentations for self-supervised learning

- **Phase 2 – Self-Supervised Learning**
  - Contrastive learning framework
  - Neural encoder trained to bring augmented views of the same audio closer in embedding space

- **Phase 3 – Audio Retrieval**
  - Embedding database construction
  - Cosine similarity–based retrieval to identify the closest matching track

- **Phase 4 – Embedding Analysis**
  - PCA and t-SNE visualizations to study embedding structure

- **Phase 5 – Qualitative Evaluation**
  - Retrieval behavior analysis on representative samples

## Repository Structure
- `submission.py` – Inference and retrieval script used for evaluation  
- `Impulse_2026.ipynb` – Complete development notebook with experiments and visualizations  
- `outputs.csv` – Test output file submitted for evaluation  
- `requirements.txt` – Required Python dependencies  

## Dataset
We use the **Free Music Archive (FMA)** dataset for training and evaluation.  
Due to size constraints, the dataset is not included in this repository.

## Key Highlights
- Fully self-supervised (no labeled training data)
- Scalable retrieval pipeline
- Clear separation between training, analysis, and inference
- Reproducible and modular design

---

