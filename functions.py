import pickle
from itertools import combinations
from multiprocessing import Lock, Pool
from operator import mul

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


def to_coords(params):
    """
    X position is peak energy
    Y position is log of peak height
    """
    coords = np.array(
        [[center, height] for [center, height, std, offset, slope] in params]
    )
    coords = coords[np.argsort(coords[:, 0])]

    return coords


@jit(nopython=True)
def linear_guass(x, center, height, std, base_offset, base_slope):
    gaussian = np.e ** -((x - center) ** 2 / (2 * std**2)) * height
    linear_base = (x - center) * base_slope + base_offset
    return gaussian + linear_base


def normal_binify(data, bins_to_median=2500):
    hist, bins = np.histogram(data, int(bins_to_median * data.max() / np.median(data)))
    return hist, (bins[:-1] + bins[1:]) / 2


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


def find_peaks_worker(wid, q, params_lock, print_lock):
    with print_lock:
        print(f"Worker #{wid} starting")

    while True:
        job = q.get()
        if job == "finish":
            with print_lock:
                print(f"Worker #{wid} done")
            q.task_done()
            return
        else:
            with print_lock:
                print(f"Processing channel {job}")

        values = channel_data.loc[job, "values"]
        hist, bins = normal_binify(values)
        params, _ = my_peaks(hist, bins, n=10)

        with params_lock:
            write_row("params.csv", "a", [job, *np.array(params).flatten()])
        with print_lock:
            print(f"Found params for channel {job}")

        q.task_done()


def find_matches_worker(find_matches_of, wid, q, match_lock, print_lock):
    with print_lock:
        print(f"Worker #{wid} starting")

    while True:
        job = q.get()
        if job == "finish":
            with print_lock:
                print(f"Worker #{wid} done")
            q.task_done()
            return
        else:
            with print_lock:
                print(f"Processing channel {job}")

        params = all_params.loc[job]

        possible_choices = []
        differences = []
        bs = []
        for choice in combinations(range(len(params)), r=find_matches_of.shape[0]):
            coords = to_coords(params[i] for i in choice)
            [energy_correct], _ = curve_fit(
                mul, coords[:, 0], find_matches_of[:, 0], p0=[1]
            )
            [height_correct], _ = curve_fit(
                mul, coords[:, 1], find_matches_of[:, 1], p0=[1]
            )
            b = [energy_correct, height_correct]
            differences.append(((find_matches_of - coords * b) ** 2).sum(axis=1).sum())
            possible_choices.append(choice)
            bs.append(b)

        best = np.argmin(differences)
        with match_lock:
            write_row(
                "matches.csv",
                "a",
                [job, *bs[best], *possible_choices[best]],
            )
        with print_lock:
            print(f"Found matches for channel {job}")

        q.task_done()


def calculate_master_coords(channel_data, channel):
    try:
        with open(f"chan{channel}_top5.pickle", "rb") as f:
            return to_coords(pickle.load(f))
    except FileNotFoundError:
        print(f"Looking for five peaks in channel {channel}")
        values = channel_data.loc[channel, "values"]
        hist, bins = normal_binify(values)
        top, _ = my_peaks(hist, bins, n=5)

        with open(f"chan{channel}_top5.pickle", "wb") as f:
            pickle.dump(top, f)

        print(f"Done finding peaks in {channel}")
        return to_coords(top)


def load_channel_data():
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

    """
    fig, axes = plt.subplots(nrows=2)
    for ax, title in zip(axes, ["Unfiltered", "Filtered"]):
        channel_data.plot.scatter(x="median", y="count", ax=ax)
        ax.set_title(title)
        channel_data = channel_data.query("(count > 50000) & (median > 4000)")
    fig.set_size_inches(4, 6)
    fig.tight_layout()
    fig.savefig("filter-show.png")
    """

    return channel_data


def big_math(channel_data, find_matches_of):
    print_lock = Lock()
    with Pool() as p:
        p.map(lambda values: my_peaks(*normal_binify(values)), channel_data["values"])

    with print_lock:
        print(f"{nworkers} workers started")

    for job in channel_data.index:
        q.put(job)

    q.join()

    for _ in range(nworkers):
        q.put("finish")

    q.join()

    print("All done")


if __name__ == "__main__":
    channel_data = load_channel_data()
    # big_math(channel_data, calculate_coords(channel_data, 1))
