# Metrics Module
This module provides a set of functions to retrieve the vocal metrics. It is designed to be easy to use and integrate into an API.

## Basic Terminology

This module base is `librosa` python library, so it is recommended to be familiar with its terminology. 
As a simple resume, here are some basic terms used in this module:

### Key Terms
- **sr**: Sample Rate, the number of samples of audio carried per second, measured in Hz or kHz.
- **hop_length**: The number of samples between successive frames, used in short-time Fourier transform (STFT) calculations.
- **f0**: Fundamental Frequency, the lowest frequency of a periodic waveform, representing the pitch of the sound.
- **voiced_flag**: A binary indicator that specifies whether a frame of audio is voiced (contains pitch) or unvoiced (does not contain pitch).
- **voiced_probs**: The probability that a given frame of audio is voiced, typically ranging from 0 to 1.

### Metrics

- **time_s**: Time in seconds for each frame.
- **frequency_hz**: Estimated fundamental frequency per frame (Hz).
- **voiced_prob**: Probability that each frame contains voice.
- **rms**: Root Mean Square energy per frame.
- **centroid_hz**: Spectral centroid per frame (Hz).
- **bandwidth_hz**: Spectral bandwidth per frame (Hz).
- **flatness**: Spectral flatness per frame.
- **rolloff_hz**: Spectral rolloff per frame (Hz).
- **zcr**: Zero-crossing rate per frame.

*You can find more information in the [librosa documentation](https://librosa.org/doc/latest/index.html).*