import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_experiment_subplot(df, experiment_name, color, ax, y_column, title):
    sns.lineplot(data=df, x=df.index, y=y_column, label=title, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(y_column)
    ax.legend()

def plot_histogram_subplot(df, experiment_name, color, ax):
    sns.histplot(data=df, x='eprewmean', bins=30, kde=True, color=color, ax=ax, label=experiment_name)
    ax.set_title(experiment_name)  # Set subtitle as experiment name
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.legend()

def plot_subplots(df_list, experiment_names, colors, y_columns, titles):
    for df, experiment_name, color in zip(df_list, experiment_names, colors):
        num_plots = len(y_columns)

        # Adjust the layout based on the number of plots
        num_rows = (num_plots + 1) // 2
        num_cols = 2
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharex=True)

        # Flatten the axes array to simplify indexing
        axes = axes.flatten()

        # Plot each variable on a separate graph
        for i, (y_column, title) in enumerate(zip(y_columns, titles)):
            plot_experiment_subplot(df, experiment_name, color, axes[i], y_column, title)
            axes[i].set_xlim(0, df.index.max())  # Set x-axis limit based on the variable's range

        # Adjust the layout
        plt.suptitle(experiment_name, fontsize=16)  # Set title as experiment name
        plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5, w_pad=0.5)  # Increase vertical and horizontal spacing
        plt.show()

def plot_histograms(df_list, experiment_names, colors):
    num_experiments = len(df_list)

    # Adjust the layout based on the number of experiments
    num_rows = (num_experiments + 1) // 2
    num_cols = 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    # Plot histogram for each experiment
    for i, (df, experiment_name, color) in enumerate(zip(df_list, experiment_names, colors)):
        plot_histogram_subplot(df, experiment_name, color, axes[i])
        axes[i].set_xlim(0, df['eprewmean'].max())  # Set x-axis limit based on the histogram's range
        axes[i].set_xlabel('')

    # Adjust the layout for the histogram graph
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=5, w_pad=0.5)  # Increase vertical and horizontal spacing
    plt.show()

if __name__ == "__main__":
    ppo2_folders = [
        'ppo2_cnn_1e6_nohposreward',
        'ppo2_cnn_1e5',
        'ppo2_cnn_1e6',
        'ppo2_cnn_1e6_hposreward1',
        'ppo2_cnn_1e6_nohposreward_20lifes',
        'ppo2_cnn_2e6',
        'ppo2_cnn_2e6_penalities',
        'ppo2_cnn_2e6_wrongpenalties'
    ]

    colors = sns.color_palette("husl", len(ppo2_folders))
    df_list = []
    experiment_names = []

    for experiment_folder, color in zip(ppo2_folders, colors):
        folder_path = os.path.join("experiments", experiment_folder)
        progress_file = os.path.join(folder_path, "progress.csv")

        if os.path.exists(progress_file):
            df = pd.read_csv(progress_file)
            df_list.append(df)
            experiment_names.append(experiment_folder)

    variable_titles = [
        'Entropy Curves',
        'Episode Reward Curves',
        'Episode Length Curves',
        'Clip Fraction Curves',
        'Value Loss Curves',
        'Explained Variance Curves',
    ]

    plot_subplots(df_list, experiment_names, colors, ['loss/policy_entropy', 'eprewmean', 'eplenmean', 'loss/clipfrac', 'loss/value_loss', 'misc/explained_variance'], variable_titles)
    plot_histograms(df_list, experiment_names, colors)
