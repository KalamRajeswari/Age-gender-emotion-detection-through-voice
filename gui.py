import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# Load models
custom_objects = {"mse": keras.losses.MeanSquaredError()}
gender_model = tf.keras.models.load_model("gender_cnn_model.h5",custom_objects=custom_objects)
age_model = tf.keras.models.load_model("age_prediction_model.h5",compile=False)
age_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
emotion_model = tf.keras.models.load_model("speech_emotion_model.h5",custom_objects=custom_objects)

def preprocess_audio(file_path, sr=22050, n_mels=128, max_duration=3):
    """Load and preprocess an audio file into a Mel spectrogram for model prediction."""
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=max_duration)
        y = librosa.util.fix_length(y, size=sr * max_duration)  # Pad or truncate

        # Convert to Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Reshape for model input (128, time_steps, 1)
        mel_spec_db = mel_spec_db[..., np.newaxis]

        return np.array([mel_spec_db])  # Add batch dimension
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
def extract_features(file_path, max_pad_len=188, num_mfcc=60):
    feature = None  # Initialize feature to avoid reference errors
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

        # Ensure both features have the same length
        min_length = min(mfccs.shape[1], mel_spec.shape[1], max_pad_len)
        mfccs = mfccs[:, :min_length]
        mel_spec = mel_spec[:, :min_length]

        # Pad if necessary
        pad_width_mfcc = max(0, max_pad_len - mfccs.shape[1])
        pad_width_mel = max(0, max_pad_len - mel_spec.shape[1])

        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width_mfcc)), mode='constant')
        mel_spec = np.pad(mel_spec, pad_width=((0, 0), (0, pad_width_mel)), mode='constant')

        feature = np.vstack((mfccs, mel_spec))[:, :128]  # Trim to (188, 128)
        print(f"Processed {file_path}: Shape {feature.shape}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return feature  # Ensures function always returns something (even if None)

emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }

def predict_emotion(file_path):
    feature = extract_features(file_path)
    if feature is not None:
        feature = np.expand_dims(feature, axis=[0, -1])  # Reshape to match model input
        prediction = emotion_model.predict(feature)
        predicted_emotion = list(emotions.values())[np.argmax(prediction)]
        print(f"Predicted emotion for {file_path}: {predicted_emotion}")
        return {predicted_emotion}
    else:
        print(f"Could not extract features for {file_path}")


# Function to extract Mel-Spectrogram
def extract_mel_spectrogram(file_path, sr=16000, duration=3, n_mels=128):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)  # Load WAV/MP3
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB

        # Normalize spectrogram
        mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

        # Resize to match model input
        mel_spec_db = np.resize(mel_spec_db, (128, 128))
        
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def predict_age(audio_file):
    mel_spec = extract_mel_spectrogram(audio_file)
    
    if mel_spec is not None:
        mel_spec = np.array(mel_spec).reshape(1, 128, 128, 1)  # Reshape for model
        predicted_age =age_model.predict(mel_spec)[0][0]  # Get prediction
        
        print(f"🔹 Predicted Age for {audio_file}: {predicted_age:.2f} years")
        return predicted_age
    else:
        print(" Error: Could not process audio file")
        return None

def predict():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if not file_path:
        return
    
    # Preprocess the audio
    features = preprocess_audio(file_path)
    
    # Predict gender
    gender_pred = gender_model.predict(features)
    gender = "Male" if gender_pred < 0.5 else "Female"
    print(gender)
    
    if gender == "Female":
        result_label.config(text="You are female,Upload the male voice",fg="white", bg="red")
        return
    
    # Predict age
    predicted_age = predict_age(file_path)
    age=int(predicted_age)
    print(f"Predicted Age: {predicted_age}")
    
    if age > 60:
        # Predict emotion for senior citizens
        detected_emotion=predict_emotion(file_path)
        result_label.config(text=f"Age: {age} (Senior Citizen)\n Gender:{gender}\nEmotion: {detected_emotion}",fg="black", bg="white")
    else:
        result_label.config(text=f"Age: {age}\n Gender:{gender}",fg="black", bg="white")

# GUI setup
root = tk.Tk()
root.title("Voice Analysis")

tk.Label(root, text="Upload a voice file for analysis:").pack(pady=10)
select_button = tk.Button(root, text="Select File", command=predict)
select_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
#arctic_a0001.wav