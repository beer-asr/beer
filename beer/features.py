
'''Basic speech features extraction.'''


from functools import lru_cache
import numpy as np
import scipy.signal


def hz2mel(freq_hz):
    'Convert Hertz to Mel value(s).'
    return 2595 * np.log10(1 + freq_hz / 700.)

def mel2hz(mel):
    'Convert Mel value(s) to Hertz.'
    return 700*(10**(mel/2595.0)-1)


def hz2bark(freq_hz):
    'Convert Hertz to Bark value(s).'
    return (29.81 * freq_hz) / (1960 + freq_hz) - 0.53


def bark2hz(bark):
    'Convert Bark value(s) to Hertz.'
    return (1960 * (bark + .53)) / (29.81 - bark - .53)


def __triangle(center, start, end, freqs):
    'Create triangular filter.'
    slopes = 1. / (center - start), 1./ (end - center)
    retval = np.zeros_like(freqs).astype(np.float)
    idxs = np.logical_and(freqs >= start, freqs <= center)
    retval[idxs] = np.linspace(slopes[0] * (freqs[idxs][0] - start),
                               slopes[0] * (freqs[idxs][-1] - start),
                               len(freqs[idxs]))
    idxs = np.logical_and(freqs >= center, freqs <= end)
    retval[idxs] = np.linspace(slopes[1] * (end - freqs[idxs][0]),
                               slopes[1] * (end - freqs[idxs][-1]),
                               len(freqs[idxs]))
    return retval


@lru_cache(maxsize=8)
def create_fbank(nfilters, fft_len=512, srate=16000, lowfreq=0, highfreq=None,
                 hz2scale=hz2mel, scale2hz=mel2hz, align_filt_center=True):
    '''Create a set of triangular filter.

    Args:
        nfilter (int): Number of filters.
        fft_len (int): Number of points of the FFT transform.
        srate (int): Sampling rate of the signal to be filtered.
        lowfreq (float): Global cut off frequency (Hz).
        highfreq (float): Global cut off frequency (Hz).
        hz2scale (function): Conversion from Hertz to the 'perceptual' scale to use.
        scale2hz (function): Inversion function of ``hz2scale``.
        align_filt_center (boolean): Align the center of the filters to FFT
            frequency bin. Set to False for exact HTK FBANK features.

    Returns
        (numpy.ndarray): The filters organized as a matrix.

    '''
    highfreq = highfreq or srate / 2
    low = hz2scale(lowfreq)
    high = hz2scale(highfreq)
    centers = np.linspace(low, high, nfilters + 2)
    if align_filt_center:
        centers = np.floor(fft_len * scale2hz(centers) / srate)
    else:
        centers = fft_len * scale2hz(centers) / srate
    filters = np.zeros((nfilters, fft_len // 2))
    for i in range(1, nfilters + 1):
        filters[i - 1, :] = __triangle(centers[i], centers[i - 1],
                                       centers[i + 1],
                                       np.arange(0, fft_len // 2))
    return filters


def add_deltas(fea, winlens=(2, 2)):
    '''Add derivatives to features (deltas, double deltas, triple_delas, ...)

    Args:
        fea (numpy.ndarray): Features matrix.
        winlens: tuple with window lengths for deltas, double deltas, ... default is (2,2)

    Returns:
        numpy.ndarray: Feature array augmented with derivatives.

    '''
    fea_list = [fea]
    for wlen in winlens:
        dfilter = -np.arange(-wlen, wlen+1)
        dfilter = dfilter / (2 * dfilter.dot(dfilter))
        fea = np.r_[fea[[0]].repeat(wlen, 0), fea, fea[[-1]].repeat(wlen, 0)]
        fea = scipy.signal.lfilter(dfilter, 1, fea, 0)[2*wlen:]
        fea_list.append(fea)
    return np.hstack(fea_list)


def fbank(signal, flen=0.025, frate=0.01, hifreq=8000, lowfreq=20, nfilters=26,
          preemph=0.97, srate=16000):
    '''Extract the FBANK features.

    The features are extracted according to the following scheme:

                    ...
                   -> Filter
        x -> |FFT| -> Filter -> log
                   -> Filter
                    ...

    The set of filters is composed of triangular filters equally spaced
    on some 'perceptual' scale (usually the Mel scale).

    Args:

        signal (numpy.ndarray): The raw audio signal.
        outdir (string): Output of an existing directory.
        fduration (float): Frame duration in seconds.
        frate (int): Frame rate in Hertz.
        hz2scale (function): Hz -> 'scale' conversion.
        nfft (int): Number of points to compute the FFT.
        nfilters (int): Number of filters.
        postproc (function): User defined post-processing function.
        srate (int): Expected sampling rate of the audio.
        scale2hz (function): 'scale' -> Hz conversion.
        srate (int): Expected sampling rate.
        window (function): Windowing function.

    '''
    # Convert the frame rate/length from second to number of samples.
    frate_samp = int(srate * frate)
    flen_samp = int(srate * flen)

    # Compute the number of frames.
    nframes = (len(signal) - flen_samp) // frate_samp + 1

    # Pre-emphasis filtering.
    s_t = np.array(signal, dtype=np.float32)
    s_t -= preemph * np.r_[s_t[0], s_t[:-1]]

    # Extract the overlapping frames.
    isize = s_t.dtype.itemsize
    sframes = np.lib.stride_tricks.as_strided(s_t, shape=(nframes, flen_samp),
                                              strides=(frate_samp * isize,
                                                       isize),
                                              writeable=False)

    # Apply the window function.
    frames = sframes * np.hamming(flen_samp)[None, :]

    # Compute FFT.
    fft_len = int(2 ** np.floor(np.log2(flen_samp) + 1))
    magspec = np.abs(np.fft.rfft(frames, n=fft_len, axis=-1)[:, :-1])

    # Filtering.
    filters = create_fbank(nfilters, fft_len, lowfreq=lowfreq, highfreq=hifreq)
    melspec = magspec @ filters.T

    return np.log(melspec + 1e-30)
