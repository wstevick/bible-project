# Wesley Stevick Problem Set 5

November 30th

Code [on GitHub](https://github.com/wstevick/bible-project). Run
```bash
python3 -m pip install -r pyproject.toml
# first run generates peaks, then fails
python3 PS5.py
# second run uses generated peaks
python3 PS5.py
```

This takes less than an hour on my computer. Saves output to `channel_data.pickle`, and generate a butt-ton of `.png` files (all untracked).

## 1-2
First, all channels with a median value less than 4000 or with fewer than 50,000 counts are dropped (see `filter-show.png`). Then, 1 and 2 are accomplished by the `my_peaks` and `find_matches` functions. Code tries to pick peaks by looking at the summed distance between each matched peak and the corresponding master peak ("coords" method), and by looking at the sum-squared difference between the histogram of the master channel and the histogram of the channel in question after splining ("shape" method). For both methods, a linear "energy correction" and "height correction" transformation is applied to minimize the difference between candidate and master peaks, to account for gain and counts differences between channels.

Master peaks are in `master_peaks.png`. The result of matching peaks with the two methods can be seen in `all-spectra-coords-aligned.png` and `all-spectra-shape-aligned.png`. The former method seemed to produce consistantly better outputs. Channels 23 and 27 didn't match correctly, so were removed (see `all-spectra-coords-aligned-nooutliers.png`).

## 3-6
See `coadded.png` for the co-added spectra after dropping deviant channels and aligning the rest. `biggest_peak_snr.png` has a histogram of the SNRs of every channel for the highest master peak. `little_peak_snr.png` has a histogram of the SNRs of every channel the peak whose height closest corresponds to 1/10th the height of the most prominent peak on the co-added spectrum. The instructions said to do this with a peak 1/100th the height, but that was a barely visible blip against neighbor peaks. In both cases, the SNR of the peak on the co-added spectrum was so much less than the SNR of the peak on the channels that including it made the bins too narrow to be visible, so was included in the axes title. On both plots also see the fit of the relevent peak on the co-added spectrum.

## 7
Considered performing DTW with the peaks using `mvmStepPattern.T()`, as this forces each peak in the reference (each master peak) to be matched exactly once by a peak in the query. However, this algorithm is essentially the same as "coords". If I have a few spare hours I'll give DTW a shot on the binned data prior to peak-finding, however I don't expect this to outperform the current method and as I intend to use DTW for drift correction on the project I'll prioratize that with my time.

# AI use
- https://chatgpt.com/share/692cbb6d-9e70-8002-a9b5-5e52aacaf200 (contains hallucination, don't use)
- https://chatgpt.com/share/692cbb8a-e7a8-8002-a6b1-8396e13f8907 (didn't use)
- https://chatgpt.com/share/692cbb98-8948-8002-a996-d7c7458428d7
