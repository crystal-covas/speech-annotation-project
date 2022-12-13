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

### Mel-spectrogram and MFCC ###
def freq_to_mel(freq):
    mel_scale=2595*np.log10(1+freq/700)
    return mel_scale

def mel_to_freq(mel):
    freq_scale=700 * (10**(mel/2595) - 1)
    return freq_scale

def mel_filterbank(n_filters,sample_rate,n_fft):
    """
    Create the mel_filterbank
    _____
    :in:
    n_filters: The total number of filters
    _____
    :out:
    array of the mel_filter_bank
    """

    #frequency range
    low_freq = 0
    high_freq = sample_rate//2

    #mel-range, from frequency to mel
    min_mel=freq_to_mel(low_freq)
    max_mel=freq_to_mel(high_freq)

    #create the x-axis of mel_filterbank
    mel_pts=np.linspace(min_mel,max_mel,2+n_filters)

    #convert from mel to frequency
    freq_pts=mel_to_freq(mel_pts)

    #freq to fft_bins conversion
    #fft_bins=(((1+n_fft)*freq_pts)//sample_rate).astype(np.int32)
    fft_bins=(((1+n_fft)*freq_pts)//sample_rate).astype(np.int32)

    mel_filter_bank=np.zeros((n_filters,1+n_fft//2))
    for i in range(1,n_filters+1):
        f_m_minus=fft_bins[i-1]
        f_m=fft_bins[i]
        f_m_plus=fft_bins[i+1]

        for j in range(f_m_minus,f_m):
            mel_filter_bank[i-1,j]=(j-fft_bins[i-1])/ (fft_bins[i] - fft_bins[i - 1])

        for j in range(f_m,f_m_plus):
            mel_filter_bank[i-1,j]=(fft_bins[i + 1] - j) / (fft_bins[i + 1] - fft_bins[i])


    mel_filter_bank=mel_filter_bank.T

    return mel_filter_bank


def make_mel_spectrogram(segments,sample_rate,n_fft,n_filters,thres_db):
    """
    Create the mel-spectrogram; the dot product between the mel_filter_bank and the spectrogram
    _____
    :in:
    mel_filterbank
    spectrogram
    _____
    :out:
    array of mel_spectrogram
    """

    #short-term-fourier-transform
    fft_segments,freq_bins = STFT(segments=segments,sample_rate=22050,n_fft=2048)

    # create the spectrogram
    spec=make_spectrogram(fft_segments,n_fft,thres_db,sample_rate)

    #create the mel-filter-bank
    mel_filter=mel_filterbank(n_filters,sample_rate,n_fft)

    # project the spectrogram onto the mel-basis
    proj_mel=np.dot(spec,mel_filter)

    mel_spec=proj_mel.T

    return mel_spec
