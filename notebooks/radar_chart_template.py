import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_radar(df: pd.DataFrame, title: str = "Radar chart", r_ticks=None, r_label: str = ""):
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns: a label column and one series.")
    labels = df.iloc[:, 0].astype(str).tolist()
    series_names = df.columns[1:].tolist()
    values = df.iloc[:, 1:].astype(float).to_numpy()
    labels_closed = labels + [labels[0]]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    if r_ticks is not None:
        ax.set_rticks(r_ticks)
    if r_label:
        ax.set_rlabel_position(0)
        ax.set_ylabel(r_label)
    for i, name in enumerate(series_names):
        vals = values[:, i].tolist()
        vals_closed = vals + [vals[0]]
        ax.plot(angles_closed, vals_closed, linewidth=2)
        ax.fill(angles_closed, vals_closed, alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=20)
    ax.grid(True)
    ax.legend(series_names, loc="upper right", bbox_to_anchor=(1.2, 1.15))
    plt.show()

if __name__ == "__main__":
    # Replace 'radar_template.csv' with your own CSV path.
    df = pd.read_csv("radar_chart.csv")
    plot_radar(df, title="Radar from CSV", r_ticks=[0.2,0.4,0.6,0.8,1.0], r_label="Score")
