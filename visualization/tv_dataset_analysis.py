import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the audio files
dir_path = "G:\Workspace\SocialBit\Data\TV detector dataset\wav"

# Define the categories
categories = ["1", "2", "3"]

# Define the microphones
microphones = ["ipad", "iphone", "pixel", "watch"]

# Define the speakers
speakers = ["jbl", "ipad", "pc", "pixel", "NA", "iphone"]

# Define a function to extract the relevant information from the filename
def get_info_from_filename(filename):
    parts = filename.split("-")
    class_label = int(parts[0])
    microphone = parts[1]
    speaker = parts[2]
    if speaker == "0":
        speaker = "NA"
    category = parts[3].split(".")[0]
    return class_label, microphone, speaker, category


# Iterate over the audio files and extract relevant information
data = []
microphone_sample_rates = {}
for filename in os.listdir(dir_path):
    if filename.endswith(".wav"):
        filepath = os.path.join(dir_path, filename)
        class_label, microphone, speaker, category = get_info_from_filename(filename)
        duration = librosa.get_duration(filename=filepath)
        sr = librosa.get_samplerate(filepath)
        microphone_sample_rates[microphone] = sr
        data.append((filepath, class_label, microphone, speaker, category, duration))

# Create a bar chart showing the distribution of data over different speakers
speaker_durations = {speaker: 0 for speaker in speakers}
for _, _, microphone, speaker, _, duration in data:
    speaker_durations[speaker] += round(duration/60, 2)
plt.bar(speaker_durations.keys(), speaker_durations.values())
plt.title('Distribution of Data over Speakers')
plt.xlabel('Speaker')
plt.ylabel('Duration (min)')
heights = [val for val in speaker_durations.values()]
for index, value in enumerate(heights):
    plt.text(index, value, str(round(value)), ha='center')
plt.show()

# Create a bar chart showing the distribution of data over different microphones
microphone_durations = {microphone: 0 for microphone in microphones}
for _, _, microphone, _, _, duration in data:
    microphone_durations[microphone] += round(duration/60, 2)
plt.bar(microphone_durations.keys(), microphone_durations.values())
plt.title('Distribution of Data over Microphones')
plt.xlabel('Microphone')
plt.ylabel('Duration (min)')
heights = [val for val in microphone_durations.values()]
for index, value in enumerate(heights):
    plt.text(index, value, str(round(value)), ha='center')
plt.show()

# Create a bar chart showing the distribution of data over different classes
class_durations = {0: 0, 1: 0}
for _, class_label, _, _, _, duration in data:
    class_durations[class_label] += round(duration/60, 2)
plt.bar(class_durations.keys(), class_durations.values())
plt.title('Distribution of Data over Classes')
plt.xlabel('Class Label')
plt.ylabel('Duration (min)')
plt.xticks([0, 1])
heights = [val for val in class_durations.values()]
for index, value in enumerate(heights):
    plt.text(index, value, str(round(value)), ha='center')
plt.show()

# Create a bar chart showing the distribution of data over different categories
category_durations = {category: 0 for category in categories}
for _, class_label, _, _, category, duration in data:
    if class_label == 1:
        category_durations[category] += round(duration/60, 2)
plt.bar(category_durations.keys(), category_durations.values())
plt.title('Distribution of Data over Categories')
plt.xlabel('Category')
plt.ylabel('Duration (min)')
heights = [val for val in category_durations.values()]
for index, value in enumerate(heights):
    plt.text(index, value, str(round(value)), ha='center')
plt.show()
