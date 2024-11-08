import os  # for file operations
import uuid  # for unique ID classifications
import librosa  # for audio feature extraction
import numpy as np  # for array manipulation
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import speech_recognition as sr  # for speech recognition
# Create an instance of the recognizer
r = sr.Recognizer()

# Emotion mapping
emotion_mapping = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}
#mel - frequency cepstral coefficients
# Extract audio features using librosa
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# # Load dataset from the directory
# def load_dataset(data_path):
#     features = []
#     labels = []
#     for folder in os.listdir(data_path):
#         folder_path = os.path.join(data_path, folder)
#         if os.path.isdir(folder_path):
#             for file in os.listdir(folder_path):
#                 if file.endswith(".wav"):
#                     file_path = os.path.join(folder_path, file)
#                     emotion = file.split("-")[2]  # Extracting the emotion part
#                     label = emotion_mapping.get(emotion, "unknown")
#                     features.append(extract_audio_features(file_path))
#                     labels.append(label)
#     return np.array(features), np.array(labels)


# Load dataset from the directory with RAVDESS filename structure
def load_dataset(data_path):
    features = []
    labels = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    parts = file.split("-")  # Split filename into parts
                    emotion_code = parts[2]  # The third part corresponds to the emotion
                    label = emotion_mapping.get(emotion_code, "unknown")  # Map to emotion label
                    features.append(extract_audio_features(file_path))
                    labels.append(label)
    return np.array(features), np.array(labels)
# Define the PyTorch model
class EmotionNet(nn.Module):
    def __init__(self, input_size):
        super(EmotionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 8)  # 8 emotion categories
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Train the model using PyTorch
def train_model(data_path):
    X, y = load_dataset(data_path)

    # Convert labels to integers using LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    # Create the model
    input_size = X_train.shape[1]
    model = EmotionNet(input_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'emotion_detector_dnn_model.pth')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = torch.argmax(y_pred, dim=1).numpy()
        print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

def predict_emotion(audio_file):
    input_size = 13  # Assuming 13 MFCC features
    model = EmotionNet(input_size)
    model.load_state_dict(torch.load('emotion_detector_dnn_model.pth'))
    model.eval()

    features = extract_audio_features(audio_file).reshape(1, -1)
    features = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(features)
        predicted_class = torch.argmax(prediction).item()
        return predicted_class

def listen_to_speech():
    with sr.Microphone() as source:
        print("Please say something")
        audio = r.listen(source)
        try:
            audio_file_path = f"audio_{uuid.uuid4()}.wav"
            with open(audio_file_path, "wb") as f:
                f.write(audio.get_wav_data())
            return audio_file_path
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error: {e}")
            return None

# Main function to train the model or predict emotion
def main():
    if not os.path.exists('/emotion_detector_dnn_model.pth'):
        print("Model Detected!")
        data_path = 'C:/Users/MuhammadTayyab/Desktop/test_to_speech/test_to_speech/Audio_Song_Actors_01-24/Actor_01'  # Replace with actual path
        print("Audio Folder found!")
        train_model(data_path)
    else:
        audio_file_path = listen_to_speech()
        if audio_file_path:
            predicted_emotion = predict_emotion(audio_file_path)
            print(f"Your current emotion is: {predicted_emotion}")

# Run the main function
if __name__ == "__main__":
    main()
