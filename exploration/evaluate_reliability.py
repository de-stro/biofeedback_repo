import numpy as np
import matplotlib.pyplot as plt

import sklearn.decomposition as skdec

with np.load('dennis.npz') as npz_file:
    eegs = npz_file['eegs'] # [ channels, samples ]
    ecg = npz_file['ecg']
    samp_rate = npz_file['sampling_rate'] # in Hz

# parameters
t_segment = 30 # in s

n_segment = int(t_segment * samp_rate) # in samples
n_signals = eegs.shape[0]

# beat template
ideal_beat = np.load('ideal_beat.npy')
ideal_beat /= np.sum(ideal_beat ** 2) # normalize

# estimate group delay of beat template (dirty hack)
noise = np.random.randn(3000)
filtered_noise = np.correlate(noise, ideal_beat, 'full')
correlation = np.correlate(filtered_noise, noise, 'full')
group_delay = np.argmax(correlation) - (len(noise) - 1)

corr_errs = 0
mse_errs = 0
total = 0
print('start:\tcorr corr(ref) | MSE MSE(ref)')
for n_start in range(0, eegs.shape[1] - n_segment, n_segment):
    total += 1

    # eeg segment
    segment = eegs[:, n_start : n_start + n_segment]

    # ecg reference signal
    reference = ecg[n_start : n_start + n_segment]
    reference /= np.sqrt(np.mean(reference ** 2))

    # ICA
    ica = skdec.FastICA()
    components = ica.fit_transform(segment.T).T

    ## evaluation

    corr = []
    corr_ref = []
    mse = []
    mse_ref = []
    for component in components:
        filtered = np.correlate(component, ideal_beat, 'full')[group_delay:group_delay+n_segment]

        corr.append(np.corrcoef(component, filtered)[0, 1])
        corr_ref.append(np.corrcoef(component, reference)[0, 1])
        mse.append(np.sum((component - filtered) ** 2))
        mse_ref.append(np.sum((component - np.sign(corr_ref[-1]) * reference) ** 2))

    idx_corr = np.argmax(corr)
    idx_corr_ref = np.argmax(np.abs(corr_ref))
    idx_mse = np.argmin(mse)
    idx_mse_ref = np.argmin(mse_ref)
    corr_ind = ' '
    if idx_corr != idx_corr_ref:
        corr_ind = '⚠'
        corr_errs += 1
    mse_ind = ' '
    if idx_mse != idx_mse_ref:
        mse_ind = '⚠'
        mse_errs += 1
    t_start = n_start / samp_rate
    print(f'{t_start}:\t{idx_corr} {idx_corr_ref} {corr_ind} | {idx_mse} {idx_mse_ref} {mse_ind}')

print(f'∑ ⚠\t{corr_errs:5d} | {mse_errs:5}  (of {total})')

