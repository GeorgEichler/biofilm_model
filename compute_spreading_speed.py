import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def central_diff(t, y):
    """
    Compute first derivatives
    """
    n = len(t)
    v = np.zeros(n)

    v[0] = (y[1] - y[0]) / (t[1] - t[0])

    for i in range(1, n-1):
        dt = t[i+1] - t[i-1]
        v[i] = (y[i+1] - y[i-1]) / dt

    # backward difference at end
    v[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return v

def fit_linear_interval(t: np.ndarray, y: np.ndarray, tmin: float, tmax: float):
    """
    Fit y ~ a * t + b on t in [tmin, tmax] via least squares.
    Returns (a, b). If interval has <2 points, falls back to using all data.
    """
    mask = (t >= tmin) & (t <= tmax)
    if mask.sum() < 2:
        mask = np.ones_like(t, dtype=bool)
    T = np.vstack([t[mask], np.ones(mask.sum())]).T
    a, b = np.linalg.lstsq(T, y[mask], rcond=None)[0]
    return float(a), float(b)

if __name__ == '__main__':
    csv_filename = "Results/data/width_evolution_g.csv"
    df = pd.read_csv(csv_filename)
    df = df.sort_values(["g", "t"]).reset_index(drop=True)
    unique_g = sorted(df["g"].unique())

    fig, ax = plt.subplots()

    for g in unique_g:
        sub = df[df["g"] == g].copy()
        t = sub["t"].to_numpy()
        w = sub["width"].to_numpy()

        v = central_diff(t, w)
        sub['v'] = v

        ax.plot(t, v, label = f"g = {g}")
        ax.set_xlabel("t")
        ax.set_ylabel("v")
        ax.legend()

    plt.show()