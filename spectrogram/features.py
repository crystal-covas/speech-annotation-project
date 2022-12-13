import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def make_segments(y,window_length,hop_length,window_name):
  """
  segmentations of the original audio y
  _____
  :in:
  y: input audio
  window_length: The length of each of the segmentation
  hop_length: The length between each window
  window_name: 5 types of window functions; hann, hamming, blackman, rect, triang
  _____
  :out:
  Segmented audios stored in a list, each element is a segmentation
  """

  window_type={'hann':signal.windows.hann(window_length),
               'hamming':signal.windows.hamming(window_length),
               'blackman':signal.windows.blackman(window_length),
               'rect':signal.windows.boxcar(window_length),
               'triang':signal.windows.triang(window_length)
               }

  segment_lst=[]
  window=window_type[window_name]
  num_segments=1+(len(y)-window_length)//hop_length
  for k in range(num_segments):
    start=k*hop_length
    end=start+window_length
    segment=y[start:end]
    segment=segment*window
    segment_lst.append(segment)
  return segment_lst


def STFT(segments,sample_rate,n_fft):
    """
    Short Term Fourier Transform of the audio segments
    _____
    :in:
    segments: The segmentation of the input audio
    sample_rate: sample rate of the audio
    n_fft: The length of the stft-ed segments
    _____
    :out:
    stft of each of the segments in a list
    """
    fft_segments=[]
    #n_fft_new=1+n_fft//2
    freq_bins=np.linspace(0,sample_rate,n_fft)
    for segment in segments:
        fft_segment=np.fft.fft(segment)
        fft_segments.append(fft_segment)
    return fft_segments,freq_bins


def make_spectrogram(stft_segments,max_freq,n_fft,sample_rate,thres_db):
    """create the spectrogram
    Creates the spectrogram array of the signal
    ------
    :in:
    stft_segments: stft-ed audio segments
    max_freq: The maximum frequency to consider
    n_fft: The length of each of stft-ed segments
    thres_db: The threshold in dB, considers signals above this threshold
    ______
    :out:
    numpy array of the spectrogram
    """
    mag_stft_segments=np.absolute(stft_segments)
    ref_mag=np.max(mag_stft_segments)

    #reference amplitude
    ref_amp=10**(thres_db/20)
    mag_stft_segments=ref_amp * (mag_stft_segments/ref_mag)

    for segment in mag_stft_segments:
        for k in range(len(segment)):
            if segment[k] < 1:
                segment[k] = 1

    spec_lst=[]
    #n_fft_new=1+n_fft//2
    max_freq_bins=(max_freq*n_fft)//sample_rate
    for mag_stft_segment in mag_stft_segments:
        spec_lst.append(20*np.log(mag_stft_segment[0:max_freq_bins]))

    return spec_lst
