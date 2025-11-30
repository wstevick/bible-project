# %% [markdown]
# # Spectral Co-Adding
#
# Name: Wesley Stevick
#
# Date: Nov 14

# %% [markdown]
# ### Peak Finding
# 1. Use the best peak-finding tools from class to find the same 5 most prominent peaks in every channel within one data file (this may involve finding more than 5 peaks and figuring out an algorithm to find which peaks should map to which.)
# 2. Fit these peaks with a Gaussian on top of a linear background.

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functions import my_peaks, normal_binify, to_coords
from scipy.interpolate import make_interp_spline

sns.set(context="notebook", style="whitegrid")

# %%
channel_data = pd.read_pickle("channel_data.pickle")
master_peaks = pd.read_pickle("chan1_top5.pickle")
find_matches_of = to_coords(master_peaks)

# %% [markdown]
# ### Traditional Analysis
# 3. Using splines with 5 peaks, co-add all the channels within one data file.
# 4. Fit the most prominent peak of each individual spectrum after scaling it. Divide the Gaussian mean by the Gaussian width ($\sigma$) and histogram this quantity (which we will refer to as the signal to noise ratio or SNR).
# 5. Add up all the spectra and fit the most prominent peak of the summed spectrum. Plot the SNR as a vertical dashed line on the SNR histogram from #2.
# 6. Repeat steps 2 and 3 for a peak that is 2 orders of magnitude smaller (i.e. 100 times less area)

# %% [markdown]
# ### DTW Analysis
# 7. Use the DTW approach on all the channels within one data file to co-add them.
# 8. Fit the most prominent peak of each individual spectrum after scaling it. Divide the Gaussian mean by the Gaussian width ($\sigma$) and histogram this quantity (which we will refer to as the signal to noise ratio or SNR).
# 9. Add up all the spectra and fit the most prominent peak of the summed spectrum. Plot the SNR as a vertical dashed line on the SNR histogram from #2.
# 10. Repeat steps 2 and 3 for a peak that is 2 orders of magnitude smaller (i.e. 100 times less area)

# %% [markdown]
# ## Side Quest -- DTW Optimization
#
# Repeat steps 7-10 and optimize the various DTW options:
# ```
# alignment_windowed = dtw(s1, s2, keep_internals=True,
#                          window_type="sakoechiba", window_args={'window_size': 2})
# ```

# %% [markdown]
# ## Side Quest -- Wavelets for Drift Correction
#
# Inverse of noise reduction. We're keeping the noise, but removing the slow time constant terms!

# %% [markdown]
# # Parts (put here in case I want to use later, not part of code)

# %%
x0 = []
windows = []
for peak in peaks:
    ([left],) = np.where(bins == peak)
    while d2y[left] < 0:
        left -= 1
    ([right],) = np.where(bins == peak)
    while d2y[right] < 0:
        right += 1
    left += int((left - right) / 2)
    right += int((right - left) / 2)
    std_guess = (bins[right] - bins[left]) / 2
    windows.append((left, right))
    baseline_slope_guess = (hist[right] - hist[left]) / 2
    baseline_offset_guess = (hist[right] + hist[left]) / 2
    [height_guess] = hist[bins == peak] - baseline_offset_guess
    x0.append(
        [peak, height_guess, std_guess, baseline_slope_guess, baseline_offset_guess]
    )

x0 = np.array(x0)
scale = x0.mean(axis=0) / 1000
x0 /= scale
x0 = x0.flatten()

# %%
from scipy.stats import norm
from numba import jit


# @jit(nopython=True, parallel=True)
def normal(x, mean, std):
    return np.e ** (-((x - mean) ** 2) / (2 * std**2)) / np.sqrt(2 * np.pi * std**2)


# @jit(nopython=True)
def error(x0, hist, bins, windows):
    calculated = np.empty(shape=hist.shape, dtype="float64")
    calculated[:] = np.nan
    a = np.arange(len(bins))
    for i, (left, right) in enumerate(windows):
        window = (a >= left) & (a <= right)
        [center, height, std, base_slope, base_offset] = x0[i]
        calculated[window & np.isnan(calculated)] = 0
        peak = normal(bins[window], center, std)
        if np.isnan(height / peak.max()):
            return 6565784574721.466 * 1000000
        peak *= height / peak.max()
        calculated[window] += peak
        calculated[window] += base_slope * (bins[window] - center) + base_offset
    difference = hist - calculated
    difference = difference[~np.isnan(difference)]
    return np.sum(difference**2) + 10000 * (x0[3, :] ** 2).sum(), calculated


def make_error_function(hist, bins, windows, scale):
    return lambda x0: error(x0.reshape(-1, 5) * scale, hist, bins, windows)


# %%
from scipy.optimize import minimize

minimization = minimize(ferr, x0, method="Powell")

# %%
ferr = make_error_function(hist, bins, windows, scale)
err, calculated = ferr(x0)

# %%
minimization

# %%
x0 = minimization.x

# %%
fig, axes = plt.subplots(nrows=5, ncols=2)
axes = axes.flatten()
for i in range(10):
    [center, height, std, base_slope, base_offset] = (x0.reshape(-1, 5) * scale)[i]
    std = min(std, 0.005)
    left, right = windows[i]
    a = np.arange(len(bins))
    xs = bins[(a >= left) & (a <= right)]
    axes[i].set_xlim(center - 5 * std, center + 5 * std)
    axes[i].step(bins, calculated, linewidth=10, alpha=0.5)
    axes[i].plot(xs, (xs - center) * base_slope + base_offset)
    axes[i].axvline(center)
    axes[i].errorbar(center, (height + base_offset) / 2, height / 2, linewidth=10)
    axes[i].step(bins, hist)
fig.set_size_inches(8 * 2, 6 * 5)
plt.tight_layout()

# %%
histr, binsr = normal_binify(channel_data.loc[19, "values"])
hist1, bins1 = normal_binify(channel_data.loc[1, "values"])
fig, axes = plt.subplots(nrows=6, ncols=2)
for col, (bins, hist, peaks) in enumerate(
    [(bins1, hist1, find_matches_of), (binsr, histr, matches[0])]
):
    for i in range(6):
        axes[i, col].step(bins, hist, where="mid", alpha=0.2)
        axes[i, col].set_yscale("log")
        if col == 1:
            axes[i, col].scatter(
                coords19[:, 0], coords19[:, 1] * 10, color="green", alpha=0.2
            )
        if i == 0:
            continue
        peak = peaks[i - 1]
        axes[0, col].scatter(peak[0], peak[1] * 10, color="red")
        axes[i, col].scatter(peak[0], peak[1] * 10, color="red")
        axes[i, col].set_xlim(peak[0] - 0.3, peak[0] + 0.3)
fig.set_size_inches(2 * 6, 6 * 4)
plt.tight_layout()

# %%
values = channels[1]
hist, bins = normal_binify(values, bins_to_median=3000)
master_peaks, windows = my_peaks(hist, bins, n=5)
fig, axes = plt.subplots(nrows=3, ncols=2)
axes = axes.flatten()
axes[0].step(bins, hist, where="mid")
axes[0].set_yscale("log")
for i, ([center, height, std, base_offset, base_slope], (left, right)) in enumerate(
    zip(master_peaks, windows)
):
    axes[0].scatter(center, height + base_offset, color="red")
    axes[i + 1].step(bins, hist, where="mid")
    xs = np.linspace(bins[left], bins[right - 1])
    fit = linear_guass(xs, center, height, std, base_offset, base_slope)
    base = (xs - center) * base_slope + base_offset
    axes[i + 1].plot(xs, fit)
    axes[i + 1].plot(xs, base)
    axes[i + 1].set_xlim(center - 5 * std, center + 5 * std)
    axes[i + 1].axvline(center, color="red")
fig.set_size_inches(6 * 2, 4 * 3)
fig.tight_layout()
# fig.savefig('top5.png')
