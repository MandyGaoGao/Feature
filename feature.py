from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
from numpy import pi, convolve
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter

#utilities:
def extract_max(pitches, magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
        new_magnitudes.append(np.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

def smooth(x, window_len=11, window='hanning'):
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def pitch_detect(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    shape = pitches.shape
    nb_samples = shape[0]
    nb_windows = shape[1]
    pitches, magnitudes = extract_max(pitches, magnitudes, shape)
    pitches = smooth(pitches, window_len=5)
    return pitches

def harmonic_extract(y, sr):
    h_range = [1]
    S = np.abs(librosa.stft(y))
    fft_freqs = librosa.fft_frequencies(sr=sr)
    S_harm = librosa.interp_harmonics(S, fft_freqs, h_range, axis=0)[0]
    return S_harm

def a_weighting_coeffs_design(sample_rate):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    numerators = [(2*pi*f4)**2 * (10**(A1000 / 20.0)), 0., 0., 0., 0.];
    denominators = convolve(
        [1., +4*pi * f4, (2*pi * f4)**2],
        [1., +4*pi * f1, (2*pi * f1)**2]
    )
    denominators = convolve(
        convolve(denominators, [1., 2*pi * f3]),
        [1., 2*pi * f2]
    )
    return bilinear(numerators, denominators, sample_rate)

def AweightPower_extract(y, sr):
    b, a = a_weighting_coeffs_design(sr)
    k = lfilter(b, a, y)
    a_weighted_power=librosa.feature.rms(y=k)
    return a_weighted_power


#extract features for each dimension:

def extract_features_vocal(p):#(24,31)   
    y,sr=librosa.load(p)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return [
        librosa.feature.zero_crossing_rate(y_harmonic),#(1,31)
        librosa.feature.spectral_centroid(y=y, sr=sr),#(1,31)
        librosa.feature.spectral_rolloff(y=y, sr=sr),#(1,31)
        librosa.feature.spectral_contrast(y=y_harmonic,sr=sr,n_bands=6),#(7,31)
        librosa.feature.spectral_flatness(y=y_harmonic),#(1,31)
        librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)#(13,31)
    ]
    
def extract_features_breath(p):#(3,31)
    y,sr=librosa.load(p)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    mfcc=librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)[12]
    return [
        mfcc,#(1,31)
        librosa.feature.delta(mfcc),#(1,31)
        librosa.feature.delta(librosa.feature.rms(y=y))#(1,31)
    ]
    
def extract_features_rhyme(p):#(385,31)
    y,sr=librosa.load(p)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return [
        librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hop_length),#(384,31)
        librosa.feature.chroma_cens(y=y_harmonic, sr=sr)#(1,31)
    ]

def extract_features_pitch(p):#(1,31)
    y,sr=librosa.load(p)
    return [
        pitch_detect(y, sr)#(1,31)     
    ]

def extract_features_emotion(p):#(1028,31)
    y,sr=librosa.load(p)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return [
        harmonic_extract(y, sr),#(1025,31)  
        AweightPower_extract(y_harmonic, sr),#(1,31)
        librosa.feature.rms(y=y),#(1,31)intensity
        librosa.feature.zero_crossing_rate(y_harmonic),#(1,31)
        #vibrato
    ]
    

#iterate to load all audio
for p in Path(r'C:\Users\gaoyu\Desktop\feature\records').glob("**/*.wav"):
    #audio_features = numpy.array(extract_features_breath(p))   #(24,31)   
    #audio_features = numpy.array(extract_features_rhyme(p))    #(3,31)
    #audio_features = numpy.array(extract_features_vocal(p))    #(385,31)
    #audio_features = numpy.array(extract_features_pitch(p))    #(1,31)
    audio_features = numpy.array(extract_features_emotion(p))   #(1028,31)
    
    feature_table = np.vstack((audio_features))
    print(feature_table.shape) 
    print(feature_table)
    
    
