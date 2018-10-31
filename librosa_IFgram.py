def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           window='hann', norm=False, center=True, ref_power=1e-6,
           clip=True, dtype=np.complex64, pad_mode='reflect'):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by [1]_.
    Calculates regular STFT as a side effect.
    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.
    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series
    sr : number > 0 [scalar]
        sampling rate of `y`
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.
    win_length : int > 0, <= n_fft
        Window length. Defaults to `n_fft`.
        See `stft` for details.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`
        See `stft` for details.
        .. see also:: `filters.get_window`
    norm : bool
        Normalize the STFT.
    center      : boolean
        - If `True`, the signal `y` is padded so that frame
            `D[:, t]` (and `if_gram`) is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` at `y[t * hop_length]`
    ref_power : float >= 0 or callable
        Minimum power threshold for estimating instantaneous frequency.
        Any bin with `np.abs(D[f, t])**2 < ref_power` will receive the
        default frequency estimate.
        If callable, the threshold is set to `ref_power(np.abs(D)**2)`.
    clip : boolean
        - If `True`, clip estimated frequencies to the range `[0, 0.5 * sr]`.
        - If `False`, estimated frequencies can be negative or exceed
          `0.5 * sr`.
    dtype : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.
    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    Returns
    -------
    if_gram : np.ndarray [shape=(1 + n_fft/2, t), dtype=real]
        Instantaneous frequency spectrogram:
        `if_gram[f, t]` is the frequency at bin `f`, time `t`
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform
    See Also
    --------
    stft : Short-time Fourier Transform
    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> frequencies, D = librosa.ifgram(y, sr=sr)
    >>> frequencies
    array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
           [  3.150e+01,   3.070e+01, ...,   1.077e+01,   1.077e+01],
           ...,
           [  1.101e+04,   1.101e+04, ...,   1.101e+04,   1.101e+04],
           [  1.102e+04,   1.102e+04, ...,   1.102e+04,   1.102e+04]])
    '''

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # Construct a padded hann window
    fft_window = util.pad_center(get_window(window, win_length,
                                            fftbins=True),
                                 n_fft)

    # Window for discrete differentiation
    # np.linspace: linearspace.
    # 0 ~ 2*π, number == n_fft, endpoint==False, retstep==False -> 0[0], 0.1[1], ....., pi-0.1[n_fft - 1] that's all.
    # devide pi into n_fft bins, and return each bin's first number
    freq_angular = np.linspace(0, 2 * np.pi, n_fft, endpoint=False) # [radian]

    d_window = np.sin(-freq_angular) * np.pi / n_fft

    stft_matrix = stft(y, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length,
                       window=window, center=center,
                       dtype=dtype, pad_mode=pad_mode)

    diff_stft = stft(y, n_fft=n_fft, hop_length=hop_length,
                     window=d_window, center=center,
                     dtype=dtype, pad_mode=pad_mode).conj()

    # Compute power normalization. Suppress zeros.
    mag, phase = magphase(stft_matrix)

    if six.callable(ref_power):
        ref_power = ref_power(mag**2)
    elif ref_power < 0:
        raise ParameterError('ref_power must be non-negative or callable.')

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = freq_angular.reshape((-1, 1)) # ndarray(n,) == (1,n)=> ndarray(n,1) like vector
    bin_offset = (-phase * diff_stft).imag / mag #(-arg(stft_matrix) * diff_stft).imag / abs(stft_matrix)

    bin_offset[mag < ref_power**0.5] = 0

    #                                       bin_offset[mag < ref_power**0.5] = 0
    if_gram = freq_angular[:n_fft//2 + 1] + bin_offset

    if norm:
        stft_matrix = stft_matrix * 2.0 / fft_window.sum()

    if clip:
        np.clip(if_gram, 0, np.pi, out=if_gram)

    if_gram *= float(sr) * 0.5 / np.pi

    return if_gram, stft_matrix
