from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL.Image import Image
from plotly.subplots import make_subplots

from lib.trainers.visualizer import VisualizeClassifier, VisualizeGenerator
from lib.types import (
    SAVE_PATHS,
    DatasetTypes,
    FileNames,
    ModelTypes,
    State,
    TrainingParameters,
    VQAParameters,
)
from lib.trainers.classification_trainer import ClassificationTrainer


def show_images_with_captions(
    images_or_paths: List[Image] | List[str], captions: List[str], cols=3
):
    """Used to display the images given as input together with the corresponding captions.

    Args:
        images_or_paths The PIL images or path to load them locally
        captions: The captions to be displayed near each image
        cols: The number of columns to display. Defaults to 3.
    """
    num_images = len(images_or_paths)
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (image_or_path, caption) in enumerate(zip(images_or_paths, captions)):
        # Check whether we have an image or a path and load adequately in both cases.
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        else:
            image = image_or_path

        ax = axes[i]

        # Display the image
        ax.imshow(image)

        # Remove the axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Add the caption
        ax.set_title(caption)

    # Remove any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def show_umap_clustering(split, dataset, num_samples=100):
    # Prepare the parameters
    train_args = VQAParameters(split=split, use_filtered_split=True)
    state = State()
    root = (
        SAVE_PATHS.BLIP2_Generator_EasyVQA
        if dataset == DatasetTypes.EASY_VQA
        else SAVE_PATHS.BLIP2_Generator_DAQUAR
    )

    parameters = TrainingParameters(
        dataset_name=dataset,
        resume_checkpoint=True,
        resume_state=False,
        model_name=ModelTypes.BLIP2Generator,
        is_trainable=False,
        train_args=train_args,
        val_args=None,
        test_args=None,
        use_wandb=False,
    )

    visualizer = VisualizeGenerator(parameters)

    try:
        history = state.load_state(root, FileNames.UMAPEmbedding.format(split))
    except Exception:
        history = visualizer.generate_embeddings()

    visualizer.display(
        history,
        num_samples=num_samples,
        filename=FileNames.UMAPClustering.format(split),
    )


def visualize_features(split, dataset):
    train_args = VQAParameters(split=split, use_filtered_split=True)
    
    parameters = TrainingParameters(
        dataset_name=dataset,
        resume_checkpoint=True,
        resume_state=False,
        model_name=ModelTypes.BLIP2Generator,
        is_trainable=False,
        train_args=train_args,
        val_args=None,
        test_args=None,
        use_wandb=False,
    )
    
    trainer = ClassificationTrainer(parameters)
    trainer.train_one_epoch(1)


def show_confusion_matrix(split, dataset):
    args = VQAParameters(split=split, use_filtered_split=True)

    parameters = TrainingParameters(
        dataset_name=dataset,
        resume_checkpoint=True,
        resume_state=True,
        model_name=ModelTypes.BLIP2Classifier,
        is_trainable=False,
        train_args=args,
        val_args=None,
        test_args=None,
        use_wandb=False,
        split=split,
    )

    visualizer = VisualizeClassifier(parameters)

    confusion_matrix = visualizer.load_confusion_matrix(split)
    visualizer.confusion_matrix(confusion_matrix)


def calculate_cardinality_and_density(dataset):
    df = dataset.raw_dataset.to_pandas()

    # Calculate Label Cardinality
    label_cardinality = df["answer"].apply(len).mean()

    # Calculate Label Density
    num_unique_labels = len(
        set([label for sublist in df["answer"] for label in sublist])
    )
    print(f"The number of unique labels: {num_unique_labels}")
    label_density = label_cardinality / num_unique_labels

    # Calculate average number of examples per sample
    avg_examples_per_sample = df["answer"].apply(len).mean()

    print(f"Label Cardinality: {label_cardinality}")
    print(f"Label Density: {label_density}")
    print(f"Average number of labels per sample: {avg_examples_per_sample:.2f}")


def calculate_label_frequency(dataset, path, multilabel=False):
    # Flatten the list of labels
    df = dataset.raw_dataset.to_pandas()
    if multilabel:
        all_labels = [label for sublist in df["answer"] for label in sublist]
    else:
        all_labels = [label for label in df["answer"]]

    # Count the frequency of each label
    label_counts = Counter(all_labels)

    # Convert to DataFrame for easier analysis
    label_distribution = pd.DataFrame.from_dict(
        label_counts, orient="index", columns=["Frequency"]
    ).sort_values(by="Frequency", ascending=False)
    
    print("Total number of items:\n", len(dataset.raw_dataset))

    # Display the top 10 most frequent labels
    print("Top 10 most frequent labels:\n", label_distribution.head(10))

    # Display the top 10 least frequent labels
    print("Top 10 least frequent labels:\n", label_distribution.tail(10))

    # Create an interactive figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Top 10 Most Frequent Labels", "Top 10 Least Frequent Labels"),
    )

    # Plot top 10 most frequent labels
    fig.add_trace(
        go.Bar(
            x=label_distribution.head(10).index,
            y=label_distribution.head(10)["Frequency"],
            name="Most Frequent",
        ),
        row=1,
        col=1,
    )

    # Plot bottom 10 least frequent labels
    fig.add_trace(
        go.Bar(
            x=label_distribution.tail(10).index,
            y=label_distribution.tail(10)["Frequency"],
            name="Least Frequent",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(height=800, width=800, title_text="Label Frequency Distribution")
    fig.write_html(f"{path}_label_frequency.html")

    # Visualize the scatter plot
    scatter_distribution = label_distribution.reset_index()
    scatter_distribution.columns = ["Label", "Frequency"]

    # Create an interactive scatter plot for the scatter plot
    fig = go.Figure(
        data=go.Scatter(
            x=list(range(len(scatter_distribution))),
            y=scatter_distribution["Frequency"],
            mode="markers",
            text=scatter_distribution["Label"],  # Add label names to the hover text
            hovertemplate="<b>Label:</b> %{text}<br><b>Frequency:</b> %{y}<extra></extra>",
            marker=dict(
                size=8,
                color=scatter_distribution["Frequency"],
                colorscale="Turbo", 
                showscale=True,
            ),
        )
    )

    fig.update_layout(
        title="Distribution of Labels",
        xaxis_title="Label Index",
        yaxis_title="Frequency",
        yaxis_type="log",
    )
    fig.write_html(f"{path}_scatter_plot.html")



def create_label_frequency_boxplot(dataset, path, multilabel=False):
    # Flatten the list of labels
    df = dataset.raw_dataset.to_pandas()
    if multilabel:
        all_labels = [label for sublist in df["answer"] for label in sublist]
    else:
        all_labels = df["answer"]

    # Count the frequency of each label
    label_counts = Counter(all_labels)

    # Convert to DataFrame for easier analysis
    all_labels = pd.DataFrame.from_dict(
        label_counts, orient="index", columns=["Frequency"]
    ).sort_values(by="Frequency", ascending=False)

    # Calculate Q1, Q3, and IQR
    Q1 = all_labels['Frequency'].quantile(0.25)
    Q2 = all_labels['Frequency'].quantile(0.5)
    Q3 = all_labels['Frequency'].quantile(0.75)
    IQR = Q3 - Q1
    
    print(f"Q1: {Q1}")
    print(f"Q2: {Q2}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")

    # Define outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Lower Whisker: {lower_bound}")
    print(f"Upper Whisker: {upper_bound}")
    
    outliers = all_labels[(all_labels['Frequency'] < lower_bound) | (all_labels['Frequency'] > upper_bound)]

    # Remove outliers from label_distribution
    label_distribution = all_labels[(all_labels['Frequency'] >= lower_bound) & (all_labels['Frequency'] <= upper_bound)]

    # Create the boxplot
    fig = go.Figure()

    fig.add_trace(go.Box(
        x0=0,  # Set x0 to 0 for the box plot
        y=label_distribution["Frequency"],
        name="Label Frequencies",
        boxpoints="all",
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(
            color="blue",
            size=4,
            line=dict(
                color="darkblue",
                width=2
            )
        ),
        line=dict(color="darkblue"),
        hoverinfo="text",
        hovertext=[f"Label: {label}<br>Frequency: {freq}" for label, freq in label_distribution.itertuples()],
    ))

    # Highlight outliers
    fig.add_trace(go.Scatter(
        x=[-0.4] * len(outliers),  # Set x to a negative value to move outliers left
        y=outliers["Frequency"],
        mode="markers",
        marker=dict(
            color="red",
            size=6,
            symbol="circle-open",
            line=dict(width=2)
        ),
        name="Outliers",
        hoverinfo="text",
        hovertext=[f"Label: {label}<br>Frequency: {freq}" for label, freq in outliers.itertuples()],
    ))

    # Update layout
    fig.update_layout(
        title="Label Frequency Distribution Boxplot",
        yaxis_title="Frequency",
        showlegend=False,
        height=600,
        width=800,
        xaxis=dict(
            range=[-0.5, 0.5],  # Set x-axis range to control positioning
            showticklabels=False,  # Hide x-axis labels
            zeroline=False,  # Hide zero line
        ),
    )

    # Use log scale for y-axis to better visualize the distribution
    fig.update_yaxes(type="log")

    # Save the figure as an interactive HTML file
    fig.write_html(f"{path}_boxplot.html")

    # Print some statistics
    print(f"Number of unique labels: {len(all_labels)}")
    print(f"Mean frequency: {all_labels['Frequency'].mean():.2f}")
    print(f"Median frequency: {all_labels['Frequency'].median():.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Total number of items: {len(dataset.raw_dataset)}")