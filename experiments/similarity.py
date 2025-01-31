import os
import sys
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Function to extract features from a .wav file
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features (e.g., MFCCs, chroma, tempo)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Aggregate features (mean across time)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    # print(mfccs_mean.shape)
    # print(chroma_mean.shape)
    
    # Ensure tempo is treated as a 1-dimensional array
    tempo_array = np.array([tempo])[0]
    # print(tempo_array.shape)
    
    # Combine features into a single vector
    feature_vector = np.hstack([mfccs_mean, chroma_mean, tempo_array])
    return feature_vector

# Function to compute similarity between two feature vectors
def compute_similarity(feature1, feature2):
    # Use cosine similarity (1 - cosine distance)
    return 1 - cosine(feature1, feature2)

# Function to analyze all .wav files in a folder and compare to a reference file
def analyze_folder_with_reference(folder_path, reference_path):
    # Extract features from the reference file
    reference_feature = extract_features(reference_path)
    
    # Get all .wav files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    similarities = []
    filenames = []
    
    # Compare each file in the folder to the reference file
    for file in files:
        file_path = os.path.join(folder_path, file)
        feature_vector = extract_features(file_path)
        similarity = compute_similarity(feature_vector, reference_feature)
        similarities.append(similarity)
        filenames.append(file)
    
    return filenames, similarities

# Function to plot the similarities as a bar plot
def plot_similarities(filenames, similarities):
    plt.figure(figsize=(10, 6))
    plt.bar(filenames, similarities, color='blue')
    plt.xlabel("Music Files")
    plt.ylabel("Similarity to Reference")
    plt.title("Similarity Between Folder Files and Reference Music")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  # Cosine similarity ranges from -1 to 1, but we limit to 0-1 for clarity
    plt.tight_layout()
    plt.show()

# Main function
def main():
    folder_path = sys.argv[2]  # Replace with your folder path
    reference_path = sys.argv[1]  # Replace with your reference file path
    
    # Analyze folder and compute similarities
    filenames, similarities = analyze_folder_with_reference(folder_path, reference_path)
    
    # Create a table of results
    results = pd.DataFrame({
        "File": filenames,
        "Similarity": similarities
    })
    print("Similarity Results:")
    print(results)
    
    # Plot the results
    plot_similarities(filenames, similarities)

if __name__ == "__main__":
    main()