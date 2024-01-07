import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_experiment_subplot(df, color, ax, y_column, title):
    sns.lineplot(data=df, x=df.index, y=y_column, label=title, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(y_column)
    ax.legend()

def plot_histogram_subplot(df, color, ax):
    sns.histplot(data=df, x='eprewmean', bins=30, kde=True, color=color, ax=ax)
    ax.set_title('Reward Distribution Histogram')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.legend()

def plot_experiment_graph(df, experiment_name, color):
    # Set up the layout for the experiment graph
    num_rows = 3
    num_cols = 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    variable_titles = [
        'Entropy Curves',
        'Episode Reward Curves',
        'Episode Length Curves',
        'Value Loss Curves',
        'Explained Variance Curves',
    ]

    # Plot each variable on a separate subplot
    for i, (y_column, title) in enumerate(zip(['policy_entropy', 'eprewmean', 'eplenmean', 'value_loss', 'explained_variance'], variable_titles)):
        plot_experiment_subplot(df, color, axes[i], y_column, title)
        axes[i].set_xlim(0, df.index.max())  # Set x-axis limit based on the variable's range

    # Plot the histogram on the last subplot
    plot_histogram_subplot(df, color, axes[-1])

    # Adjust the layout
    plt.suptitle(experiment_name, fontsize=16)  # Set title as experiment name
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.5, w_pad=0.5)  # Increase vertical and horizontal spacing
    plt.show()

if __name__ == "__main__":
    a2c_folders = [
        'a2c_cnn_2e6_finalmodel',
        'a2c_cnn_2e6_gettofinalboss',
        'a2c_cnn_2e6_nohposreward',
    ]

    colors = sns.color_palette("husl", len(a2c_folders))

    for experiment_folder, color in zip(a2c_folders, colors):
        folder_path = os.path.join("experiments", experiment_folder)
        progress_file = os.path.join(folder_path, "progress.csv")

        if os.path.exists(progress_file):
            df = pd.read_csv(progress_file)
            plot_experiment_graph(df, experiment_folder, color)
