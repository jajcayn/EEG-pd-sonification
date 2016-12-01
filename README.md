# EEG-pd-sonification
## Description
Provides analysis tools to sonificate EEG signal.

Consists of python class able to read offline data (txt, edf, [more to come]) or read online data from Emotiv Epoc device and pure data patch for generating sounds. After preprocessing and computing statistical measures of the provided signal, sends indices via OSC protocol to pure data for real-time analysis. 

At this point following preprocessing techniques are available:
* transformation to average reference
* bandpass filtering
* computation of spectral centroid
* Hjorth analysis (activity, mobility and complexity)

The patch for generating sounds (in pure data) is very primitive at this moment.

## Dependencies
### Python
* numpy - `pip install numpy`
* pyOSC - `pip install pyOSC`
* keypy - `pip install keypy`
* emokit - `pip install emokit`
* mne - for reading offline data in *.edf format - `pip install mne`
* scipy - if detrening data before filtering - `pip install scipy`

### pure data
* iemnet - [puredata.info/downloads/iemnet](https://puredata.info/downloads/iemnet)
* OSC - [puredata.info/downloads/osc](https://puredata.info/downloads/osc)
