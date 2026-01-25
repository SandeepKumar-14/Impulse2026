import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F

# paths (relative, no Colab)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

ENCODER_PATH = os.path.join(WEIGHTS_DIR, "encoder_final.pth")
DB_PATH = os.path.join(WEIGHTS_DIR, "database_embeddings.pt")

# load model (same architecture you trained)
class AudioEncoder(torch.nn.Module):
    def __init__(self, input_dim=128, emb_dim=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

model = AudioEncoder()
model.load_state_dict(torch.load(ENCODER_PATH, map_location="cpu"))
model.eval()

database = torch.load(DB_PATH, map_location="cpu")

def extract_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    feat = torch.tensor(np.mean(mfcc, axis=1)).float().unsqueeze(0)
    with torch.no_grad():
        emb = model(feat)
    return emb.squeeze(0)

def predict_track(audio_path):
    q_emb = extract_embedding(audio_path)
    best_id, best_score = None, -1
    for tid, db_emb in database.items():
        score = F.cosine_similarity(
            q_emb.unsqueeze(0),
            db_emb.unsqueeze(0)
        ).item()
        if score > best_score:
            best_score, best_id = score, tid
    return best_id, best_score
