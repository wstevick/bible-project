import operator
import pickle
from itertools import combinations
from multiprocessing import Pool

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import jit
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

# global variables used by multiple functions
channel_data = None  # pandas DataFrame - one row per channel
master_peaks = None  # (number of master peaks)x(number of parametrs) array
find_matches_of = None  # (number of master peaks)x2 array - sorted by energy
match_shape = None  # histogram values of the silhouette of the master channel
match_allbins = None  # bins of `match_shape`


# general utilify functions
def save_figure(figure, savename):
    figure.tight_layout()
    figure.savefig(savename)
    print(f"{savename!r} generated")


def to_coords(params):
    """
    Convert a peak to an xy Cartesian point
    X position is peak energy
    Y position is peak height
    """
    coords = np.array(
        [[center, height] for [center, height, std, offset, slope] in params]
    )
    coords = coords[np.argsort(coords[:, 0])]

    return coords


def normal_binify(data, bins_to_median=2500):
    """
    Histogram a dataset such that each bin is 1/`bins_to_median`th the median.
    """
    hist, bins = np.histogram(data, int(bins_to_median * data.max() / np.median(data)))
    return hist / len(data), (bins[:-1] + bins[1:]) / 2


# peak finding
@jit(nopython=True)
def linear_guass(x, center, height, std, base_offset, base_slope):
    """
    Gaussian + linear function
    """
    gaussian = np.e ** -((x - center) ** 2 / (2 * std**2)) * height
    linear_base = (x - center) * base_slope + base_offset
    return gaussian + linear_base


def my_peaks(hist, bins, n=10):
    d2y = savgol_filter(hist, 10, 3, deriv=2)
    indexes, _ = find_peaks(-d2y, height=0)
    fit_params = []
    windows = []
    for index in indexes:
        center_guess = bins[index]
        # calculate where second derivative crosses zero on left and right side of peak
        # these are around one sigma away from the center of the peak
        left_shift = 0
        while index - left_shift >= 0 and d2y[index - left_shift] < 0:
            left_shift += 1
        left = max(index - left_shift, 0)
        right_shift = 0
        while index + right_shift < len(d2y) and d2y[index + right_shift] < 0:
            right_shift += 1
        right = min(index + right_shift, len(bins) - 1)
        sigma_guess = (bins[right] - bins[left]) / 2

        # rise over run
        slope_guess = (hist[right] - hist[left]) / 2 / sigma_guess
        offset_guess = max((hist[right] + hist[left]) / 2, 0)
        height_guess = max(hist[index] - offset_guess, 0)

        # fit the section of the curve  at what we currently think is 3 sigma on each side
        window_left = max(int(index - left_shift * 3), 0)
        window_right = min(int(index + right_shift * 3) + 1, len(bins))

        if window_right - window_left >= 5:  # noqa: PLR2004
            xdata = bins[window_left:window_right]
            ydata = hist[window_left:window_right]
            maximum_height = ydata.max()
            if maximum_height == 0:
                continue
            params, _ = curve_fit(
                linear_guass,
                xdata,
                ydata,
                [
                    center_guess,
                    height_guess,
                    sigma_guess,
                    offset_guess,
                    slope_guess,
                ],
                maxfev=16000000,
                bounds=(
                    [bins[window_left], 0, 0, 0, -np.inf],
                    [
                        bins[window_right - 1],
                        maximum_height,
                        np.inf,
                        maximum_height,
                        np.inf,
                    ],
                ),
            )
            fit_params.append(params)
            windows.append((window_left, window_right))
    fit_params = np.array(fit_params)
    heights = fit_params[:, 1]
    select = np.argsort(heights)[-n:]
    return fit_params[select], np.array(windows)[select]


def find_peaks_worker(chan):
    """
    Runs in parallel on all cores
    """
    print("Finding peaks for", chan)
    values = channel_data.loc[chan, "values"]
    try:
        hist, bins = normal_binify(values)
    except ValueError:
        return None
    params, _ = my_peaks(hist, bins, n=10)
    return params


def find_matches(chan, difference_method):
    peaks = channel_data.loc[chan, "peaks"]

    possible_choices = []
    differences = []
    bs = []
    for choice in combinations(range(len(peaks)), r=find_matches_of.shape[0]):
        coords = to_coords(peaks[choice, :])
        [energy_correct], _ = curve_fit(
            operator.mul, coords[:, 0], find_matches_of[:, 0], p0=[1]
        )
        [height_correct], _ = curve_fit(
            operator.mul, coords[:, 1], find_matches_of[:, 1], p0=[1]
        )
        b = [energy_correct, height_correct]
        differences.append(difference_method(chan, coords, b))
        possible_choices.append(choice)
        bs.append(b)

    best = np.argmin(differences)
    return bs[best], peaks[possible_choices[best], :], differences[best]


def coords_difference(chan, coords, b):
    return ((find_matches_of - coords * b) ** 2).sum(axis=1).sum()


def find_coords_matches_worker(chan):
    """
    Runs in parallel on call cores
    """
    difference_method = coords_difference
    name = "coords"

    print(f"Finding {name} matches for", chan)
    best, peaks, difference = find_matches(chan, difference_method)
    return {
        f"{name}_ec": best[0],
        f"{name}_hc": best[1],
        f"{name}_diff": difference,
        f"matches_{name}": peaks,
    }


def shape_difference(chan, coords, b):
    spl = make_interp_spline([0, *coords[:, 0]], [0, *find_matches_of[:, 0]])
    values = channel_data.loc[chan, "values"]
    new_values = spl(values)
    hist, _ = np.histogram(new_values, match_allbins)
    return np.sum((hist * b[1] - match_shape) ** 2)


def find_shape_matches_worker(chan):
    """
    Runs in parallel on call cores
    """
    difference_method = shape_difference
    name = "shape"

    print(f"Finding {name} matches for", chan)
    best, peaks, difference = find_matches(chan, difference_method)
    return {
        f"{name}_ec": best[0],
        f"{name}_hc": best[1],
        f"{name}_diff": difference,
        f"matches_{name}": peaks,
    }


def align_spectra(master_centers):
    all_values = []
    return all_values


# data retrieval functions
def calculate_master_peaks(channel):
    try:
        with open(f"chan{channel}_top5.pickle", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Looking for five peaks in channel {channel}")
        values = channel_data.loc[channel, "values"]
        hist, bins = normal_binify(values)
        top, windows = my_peaks(hist, bins, n=5)
        with open(f"chan{channel}_top5_windows.pickle", "wb") as f:
            pickle.dump(bins[windows], f)

        with open(f"chan{channel}_top5.pickle", "wb") as f:
            pickle.dump(top, f)

        print(f"Done finding peaks in {channel}")
        return top


def load_channel_data(make_plot=False):
    try:
        return pd.read_pickle("channel_data.pickle")
    except FileNotFoundError:
        pass

    channel_data = pd.DataFrame(columns=["values"])

    with h5py.File("Gamma/210601_NBS295-106/20210601_152616_mass-001.hdf5") as f:
        for key in f.keys():  # noqa: SIM118
            channel_data.loc[int(key.removeprefix("chan")), "values"] = np.array(
                f[key]["filt_value"]
            )

    channel_data.sort_index(inplace=True)

    for idx, [values] in channel_data.iterrows():
        values = values[values > 0]  # noqa: PLW2901
        channel_data.loc[idx, "values"] = values[values < np.percentile(values, 99)]

    channel_data["median"] = channel_data["values"].apply(np.median)
    channel_data["count"] = channel_data["values"].apply(len)

    if make_plot:
        fig, axes = plt.subplots(nrows=2)
        for ax, title in zip(axes, ["Unfiltered", "Filtered"]):
            channel_data.plot.scatter(x="median", y="count", ax=ax)
            ax.set_title(title)
            channel_data = channel_data.query("(count > 50000) & (median > 4000)")
        fig.set_size_inches(4, 6)
        fig.suptitle("Showing which channels are trimmed before analysis")
        save_figure(fig, "filter-show.png")

    return channel_data


# plotting functions
def make_master_peaks_plot(hist, bins, savename, title, show_fit=True):
    fig, axes = plt.subplots(nrows=3, ncols=2)

    with open("chan1_top5_windows.pickle", "rb") as f:
        windows = pickle.load(f)

    peak_centers = master_peaks[:, 0]
    peak_heights = [hist[np.abs(bins - center).argmin()] for center in peak_centers]

    for aidx, ax in enumerate(axes.flatten(), start=-1):
        ax.step(bins, hist, where="mid")
        ax.scatter(peak_centers, peak_heights, color="red")
        if aidx > -1:
            [center, height, std, offset, slope] = master_peaks[aidx]
            ax.axvline(center, color="black", alpha=0.5)
            ax.set_xlim(center - 100, center + 100)
            if show_fit:
                window = windows[aidx]
                xs = np.linspace(*window)
                baseline = (xs - center) * slope + offset
                gaussian = np.e ** -((xs - center) ** 2 / (2 * std**2)) * height
                ax.plot(xs, baseline)
                ax.plot(xs, gaussian + baseline)

    fig.set_size_inches(9, 9)
    fig.suptitle(title)
    save_figure(fig, savename)


def make_waterfall_plot(name, savename, title):
    print(f"Generating waterfall plot {name!r}")
    fig, axes = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(title)

    def adjust(y):
        return y + a

    bins = None
    for aidx, ax in enumerate(axes.flatten()):
        a = 0

        if aidx == 0:
            ax.set_xlim(0, 13000)
        for channel, stepcolor in zip(
            channel_data.index,
            sns.color_palette("colorblind", n_colors=len(channel_data)),
        ):
            hc = channel_data.loc[channel, f"{name}_hc"]
            coords = to_coords(channel_data.loc[channel, f"matches_{name}"])
            spl = make_interp_spline([0, *coords[:, 0]], [0, *find_matches_of[:, 0]])
            values = channel_data.loc[channel, "values"]
            new_values = spl(values)
            if bins is None:
                hist, bins = np.histogram(new_values, 40000)
            else:
                hist, _ = np.histogram(new_values, bins)
            ax.step(
                bins[:-1],
                adjust(hist * hc),
                color=stepcolor,
                alpha=0.75,
                where="pre",
            )
            a -= 100

    for peak, ax in zip(master_peaks, axes.flatten()[1:]):
        ax.set_xlim(peak[0] - 100, peak[0] + 100)

    fig.set_size_inches(2 * 8, 3 * 12)
    save_figure(fig, savename)


def snr_plot(
    added_hist, aligned_hists, bins, peak_center, window_size, savename, nbins, title
):
    window = (peak_center - window_size < bins) & (bins < peak_center + window_size)
    xdata = bins[window]

    def fit_peak(hist):
        ydata = hist[window]
        height = ydata[int(ydata.shape[0] / 2)]
        peak, _ = curve_fit(
            lambda x, *args: linear_guass(x, peak_center, *args),
            xdata,
            ydata,
            [height, 0, 0, 1],
            maxfev=16000000,
            bounds=([0, 0, 0, -np.inf], [height, np.inf, height, np.inf]),
        )
        return peak

    snrs = [peak_center / std for hist in aligned_hists if (std := fit_peak(hist)[1])]
    [height, std, offset, slope] = fit_peak(added_hist)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    # axes[0].axvline(peak_center / std, color="black")
    axes[0].hist(snrs, bins=nbins)
    axes[0].set_title(f"SNR histogram (global SNR at {peak_center / std:.2f})")

    axes[1].step(bins, added_hist, where="mid")
    baseline = (xdata - peak_center) * slope + offset
    gaussian = np.e ** -((xdata - peak_center) ** 2 / (2 * std**2)) * height
    axes[1].plot(xdata, baseline)
    axes[1].plot(xdata, gaussian + baseline)
    axes[1].set_xlim(peak_center - 100, peak_center + 100)
    axes[1].set_title("Fit peak of co-added data")

    fig.suptitle(f"SNR for {title}")
    save_figure(fig, savename)


def main():
    global channel_data  # noqa: PLW0603
    global master_peaks  # noqa: PLW0603
    global find_matches_of  # noqa: PLW0603
    global match_shape  # noqa: PLW0603
    global match_allbins

    sns.set_theme(style="whitegrid", context="paper")

    channel_data = load_channel_data(make_plot=True)

    master_peaks = calculate_master_peaks(1)
    make_master_peaks_plot(
        *normal_binify(channel_data.loc[1, "values"]),
        "master_peaks.png",
        "Master Peaks on Channel 1",
    )
    find_matches_of = to_coords(master_peaks)
    values1 = channel_data.loc[1, "values"]
    match_shape, match_allbins = np.histogram(
        values1, bins=int(2500 * values1.max() / np.median(values1))
    )
    match_shape = match_shape / len(values1)

    with Pool() as p:
        if "peaks" not in channel_data.columns:
            channel_data["peaks"] = p.map(find_peaks_worker, channel_data.index)
            channel_data = channel_data[~pd.isna(channel_data["peaks"])]
            channel_data.to_pickle("channel_data.pickle")
        for worker, name in [
            (find_coords_matches_worker, "matches_coords"),
            (find_shape_matches_worker, "matches_shape"),
        ]:
            if name not in channel_data.columns:
                match_data = pd.DataFrame(
                    p.map(worker, channel_data.index), index=channel_data.index
                )
                channel_data = pd.concat([channel_data, match_data], axis=1)
                channel_data.to_pickle("channel_data.pickle")

    make_waterfall_plot(
        "coords",
        "all-spectra-coords-aligned.png",
        'All spectra aligned by "Coords" method',
    )
    make_waterfall_plot(
        "shape",
        "all-spectra-shape-aligned.png",
        'All spectra aligned by "shape" method',
    )
    channel_data = channel_data.drop([23, 27])  # didn't coords align
    make_waterfall_plot(
        "coords",
        "all-spectra-coords-aligned-nooutliers.png",
        'All non-deviant spectra aligned by "coords" method',
    )

    # align everything
    aligned_values = []
    for matched, values in zip(channel_data["matches_coords"], channel_data["values"]):
        spl = make_interp_spline(
            [0, *to_coords(matched)[:, 0]],
            [0, *find_matches_of[:, 0]],
        )
        new_values = spl(values)
        aligned_values.append(new_values)

    # bin everything
    bin_width = np.median(np.concat(aligned_values)) / 2500
    bins = np.arange(0, 15000, bin_width)
    nbins = len(bins) - 1
    added_hist = np.zeros(nbins)
    aligned_hists = []
    total_npoints = sum(map(len, aligned_values))
    for values in aligned_values:
        hist, _ = np.histogram(values, bins)
        hist = hist / total_npoints
        added_hist += hist
        aligned_hists.append(hist)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    make_master_peaks_plot(
        added_hist, bin_centers, "coadded.png", "Co-added peaks", show_fit=False
    )

    most_prominent = master_peaks[master_peaks[:, 1].argmax()]
    snr_plot(
        added_hist,
        aligned_hists,
        bin_centers,
        most_prominent[0],
        20,
        "biggest_peak_snr.png",
        50,
        f"Biggest Peak ({int(most_prominent[0])}arb)",
    )

    allpeaks, _ = my_peaks(added_hist, bin_centers, n=0)
    most_prominent_on_allpeaks = allpeaks[np.argmax(allpeaks[:, 1])]
    ideal_size = most_prominent_on_allpeaks[1] / 10
    little_peak = allpeaks[np.argmin(np.abs(allpeaks[:, 1] - ideal_size))]

    snr_plot(
        added_hist,
        aligned_hists,
        bin_centers,
        little_peak[0],
        20,
        "little_peak_snr.png",
        10,
        f"Little Peak ({int(little_peak[0])}arb)",
    )


if __name__ == "__main__":
    main()
