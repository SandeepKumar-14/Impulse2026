# Impulse 2026 – Self-Supervised Audio Retrieval

This repository contains our submission for the Impulse 2026 Signal Processing & Machine Learning Hackathon.

## Approach
- **Phase 1:** Audio preprocessing using MFCC features and augmentations  
- **Phase 2:** Self-supervised contrastive learning to train an audio encoder  
- **Phase 3:** Audio retrieval using cosine similarity on learned embeddings  
- **Phase 4:** Embedding analysis and visualization  
- **Phase 5:** Qualitative retrieval demonstrations  

## Files
- `submission.py` – Inference script (loads trained weights and performs retrieval)
- `Impulse_2026.ipynb` – Development notebook
- `outputs.csv` – Test output file for evaluation
- `requirements.txt` – Dependencies

## Dataset
The dataset used is the FMA (Free Music Archive) dataset.  
Due to size constraints, the dataset is not included in this repository.

