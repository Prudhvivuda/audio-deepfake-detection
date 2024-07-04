import librosa
import librosa.display
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import numpy as np



##############################################################################################################################
# Prosodic Features
# - Fundamental Frequency (F0)
# - Energy
# - Speaking Rate
# - Pauses
# - Intonation

# Function to extract F0 (Fundamental Frequency)
def extract_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0, sr=sr)
    return f0, times

# Function to extract energy
def extract_energy(y):
    energy = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(energy)
    return energy, times

# Function to extract speaking rate and pauses using Praat
def extract_speaking_rate_and_pauses(audio_file):
    snd = parselmouth.Sound(audio_file)
    total_duration = snd.get_total_duration()
    
    # Extract pitch and intensity
    pitch = snd.to_pitch()
    intensity = snd.to_intensity()
    
    # Extract syllables using intensity
    intensity_values = intensity.values.T
    threshold = 0.3 * max(intensity_values)
    syllable_count = len([1 for i in intensity_values if i > threshold])
    
    # Calculate speaking rate
    speaking_rate = syllable_count / total_duration
    
    # Extract pauses
    silences = call(snd, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.1, "silent", "sounding")
    pauses = []
    
    num_intervals = call(silences, "Get number of intervals", 1)
    
    for i in range(1, num_intervals + 1):
        interval_label = call(silences, "Get label of interval", 1, i)
        if interval_label == "silent":
            start_time = call(silences, "Get start time of interval", 1, i)
            end_time = call(silences, "Get end time of interval", 1, i)
            pauses.append((start_time, end_time))
    
    return speaking_rate, pauses

# Function to extract intonation
def extract_intonation(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    times = librosa.times_like(pitches, sr=sr)
    return pitches, times


##############################################################################################################################
# Temporal Features
# - Zero-Crossing Rate
# - Autocorrelation

# Function to extract Zero-Crossing Rate
def extract_zero_crossing_rate(y, sr):
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    times = librosa.times_like(zero_crossings[0], sr=sr)
    return zero_crossings[0], times

def extract_autocorrelation(y, sr):
    autocorr = librosa.autocorrelate(y)
    lags = np.arange(len(autocorr)) / sr
    return autocorr, lags