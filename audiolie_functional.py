from model_object import CNN
import torch
import numpy as np
import librosa


# analyze_audio function outputs confidence vector of [positive, negative, neutral]
def analyze_audio(audio_path):
    # Initialize model
    model = torch.load("3labelmodel0.pt")
    model.eval()

    # Transform audio file
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Extract features - Example features (MFCCs and Spectral Centroid)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Mel-frequency cepstral coefficients
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Spectral centroid

        # Aggregate features (mean, standard deviation, etc.)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)

        # Concatenate all features into a single array
        features = np.concatenate([mfccs_mean, mfccs_std, [centroid_mean, centroid_std]])

    except Exception as e:
        print(f"Error encountered while processing {audio_path}: {e}")
        return None

    audio_tensor = torch.tensor(features, dtype=torch.float32).view(4, 7).unsqueeze(0)
    output = model(audio_tensor)
    return output

print(analyze_audio("03-01-03-02-01-02-01.wav"))

