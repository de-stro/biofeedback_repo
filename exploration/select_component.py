import numpy as np
import matplotlib.pyplot as plt

import sklearn.decomposition as skdec

with np.load('dennis.npz') as npz_file:
    eegs = npz_file['eegs'] # [ channels, samples ]
    ecg = npz_file['ecg']
    samp_rate = npz_file['sampling_rate'] # in Hz

# parameters
t_start = 500 # in s
t_segment = 60 # in s

n_start = int(t_start * samp_rate) # in samples
n_segment = int(t_segment * samp_rate) # in samples
n_signals = eegs.shape[0]

# time (for plotting)
t = np.arange(n_segment) / samp_rate # in s

# eeg segment
segment = eegs[:, n_start : n_start + n_segment]

# ecg reference signal
reference = ecg[n_start : n_start + n_segment]
reference /= np.sqrt(np.mean(reference ** 2))

# beat template
ideal_beat = np.load('ideal_beat.npy')
ideal_beat /= np.sum(ideal_beat ** 2) # normalize

# estimate group delay of beat template (dirty hack)
noise = np.random.randn(3000)
filtered_noise = np.correlate(noise, ideal_beat, 'full')
correlation = np.correlate(filtered_noise, noise, 'full')
group_delay = np.argmax(correlation) - (len(noise) - 1)

# ICA
ica = skdec.FastICA(random_state=37)
components = ica.fit_transform(segment.T).T

# evaluation
fig, axs = plt.subplots(components.shape[0], 1, sharex=True)
for component, ax in zip(components, axs):
    ax.plot(t, component)
    filtered = np.correlate(component, ideal_beat, 'full')[group_delay:group_delay+n_segment]
    ax.plot(t, filtered)
    corrcoef_ref = np.corrcoef(component, reference)[0, 1]
    ax.plot(t, np.sign(corrcoef_ref) * reference)
    print(f'Corr: {np.corrcoef(component, filtered)[0, 1]:.2f},', end='\t')
    print(f'Corr (ref): {corrcoef_ref:.2f},', end='\t')
    print(f'MSE: {np.sum((component - filtered) ** 2):.2f},', end='\t')
    print(f'MSE (ref): {np.sum((component - np.sign(corrcoef_ref) * reference) ** 2):.2f}')

plt.show()
