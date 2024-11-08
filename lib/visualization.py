from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL.Image import Image
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
import math
import random
import torch
from lib.trainers.classification_trainer import ClassificationTrainer
import textwrap


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


def display_complete_bar_chart(dataset, path, multilabel=False, title=None):
    """Expected to replace calculate_label_frequency"""
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

    # Create an interactive figure with two subplots
    fig = make_subplots(
        rows=1,
        cols=1,
    )

    # Plot top 10 most frequent labels
    fig.add_trace(
        go.Bar(
            x=label_distribution.index,
            y=label_distribution["Frequency"],
            name="Most Frequent",
        ),
        row=1,
        col=1,
    )

    # Update layout
    fig.update_layout(height=800, width=800, title_text=title, title_x=0.5)
    fig.write_image(f"{path}_complete_label_frequency_chart.pdf")

def calculate_label_frequency(train_dataset, valid_dataset, test_dataset, path, multilabel=False, title=None):
    def process_dataset(dataset):
        if hasattr(dataset, 'raw_dataset'):
            dataset = dataset.raw_dataset
        df = dataset.to_pandas()
        if multilabel:
            all_labels = [label for sublist in df["answer"] for label in sublist]
        else:
            all_labels = [label for label in df["answer"]]
        label_counts = Counter(all_labels)
        return pd.DataFrame.from_dict(
            label_counts, orient="index", columns=["Frequency"]
        ).sort_values(by="Frequency", ascending=False)

    # Process datasets
    train_distribution = process_dataset(train_dataset)

    datasets = [(train_distribution, "Training", "royalblue")]
    
    if valid_dataset is not None:
        valid_distribution = process_dataset(valid_dataset)
        datasets.append((valid_distribution, "Validation", "darkorange"))

    
    if test_dataset is not None:
        test_distribution = process_dataset(test_dataset)
        datasets.append((test_distribution, "Test", "limegreen"))


    # Create the figure
    fig = go.Figure()

    # Add traces for each dataset
    for distribution, name, color in datasets:
        fig.add_trace(
            go.Bar(
                x=distribution.index,
                y=distribution["Frequency"],
                name=name,
                marker_color=color,
                opacity=1,
            )
        )

    # Update layout
    fig.update_layout(
        barmode='group',
        height=600,
        width=1000,
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Labels",
        yaxis_title="Frequency",
        legend_title="Dataset",
        font=dict(size=12),
        xaxis={'categoryorder':'total descending'},
        showlegend=True
    )

    # Save the figure
    fig.write_html(f"{path}_label_frequency.html")
    fig.write_image(f"{path}_label_frequency.pdf")
    fig.write_image(f"{path}_label_frequency.png")
    
    fig.show()


def visualize_scatter_plot(dataset, path, multilabel=False, title=None):
    # Flatten the list of labels
    if hasattr(dataset, 'raw_dataset'):
        dataset = dataset.raw_dataset
    else:
        dataset = dataset
    df = dataset.to_pandas()
    
    if multilabel:
        all_labels = [label for sublist in df["answer"] for label in sublist]
    else:
        all_labels = [label for label in df["answer"]]
    label_counts = Counter(all_labels)
    
    # Convert to DataFrame for easier analysis
    label_distribution = pd.DataFrame.from_dict(
        label_counts, orient="index", columns=["Frequency"]
    ).sort_values(by="Frequency", ascending=False)
    
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
    fig.write_image(f"{path}_label_frequency.svg")
    fig.write_html(f"{path}_scatter_plot.html")
    fig.write_image(f"{path}_scatter_plot.pdf")



def create_label_frequency_boxplot(dataset, path, multilabel=False, title=None):
    # Flatten the list of labels
    if hasattr(dataset, 'raw_dataset'):
        dataset = dataset.raw_dataset
    else:
        dataset = dataset
    df = dataset.to_pandas()
    if multilabel:
        all_labels = [label for labels in df["answer"] for label in labels]
    else:
        all_labels = df["answer"]

    # The output needs to be hashable due to using the Counter below
    all_labels = [tuple(label) if isinstance(label, np.ndarray) else label for label in all_labels]

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
        name="Inliers",
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
        title=title,
        title_x=0.5,
        yaxis_title="Frequency",
        showlegend=True,  # Ensure legend is shown
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

    # Save the figure as a PDF
    fig.write_html(f"{path}_boxplot.html")
    fig.write_image(f"{path}_boxplot.pdf")

    # Print some statistics
    print(f"Number of unique labels: {len(all_labels)}")
    print(f"Mean frequency: {all_labels['Frequency'].mean():.2f}")
    print(f"Median frequency: {all_labels['Frequency'].median():.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Total number of items: {len(dataset)}")
    
    fig.show()


def display_sample_images(dataset, name, path, num_images=20, font_size=8):
    # Get random indices
    indices = random.sample(range(len(dataset)), num_images)
    
    # Calculate the number of rows and columns
    cols = 5  # We'll keep 5 columns as in the original
    rows = math.ceil(num_images / cols)
    
    # Set up the plot
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Flatten axes if it's a 2D array
    axes = axes.flatten() if num_images > cols else [axes]
    
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        if i < num_images:
            example = dataset[idx]
            
            # Convert image tensor to PIL Image if necessary
            image = example['image']
            
            # Display the image
            ax.imshow(image)
            ax.axis('off')
            
            # Add question and answer as title
            question = textwrap.fill(example['question'], width=20)
            answer = example['answer']
            ax.set_title(f"Q: {question}\nA: {answer}", fontsize=font_size, wrap=True)
        else:
            # Remove unused subplots
            fig.delaxes(ax)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Add padding between rows
    
    # Save the figure as a PDF
    plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free up memory
    
    print(f"Sample images saved as {path}")

def display_class_specific_images(dataset, name, path, class_types, images_per_class=5, font_size=10):
    # Set up the plot
    rows = len(class_types)
    cols = images_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5.5*rows))  # Increased height for padding
    
    for row, class_type in enumerate(class_types):
        # Find all indices for the current class
        class_indices = [i for i, example in enumerate(dataset) if example['answer'] == class_type]
        
        # If there are fewer than 5 images for this class, use all available
        num_images = min(images_per_class, len(class_indices))
        selected_indices = random.sample(class_indices, num_images)
        
        for col, idx in enumerate(selected_indices):
            ax = axes[row, col] if rows > 1 else axes[col]
            example = dataset[idx]
            
            # Display the image
            ax.imshow(example['image'])
            ax.axis('off')
            
            # Add question and answer as title
            question = example['question']
            answer = example['answer']
            
            # Wrap the question text
            wrapped_question = textwrap.fill(question, width=20)
            
            ax.set_title(f"Q: {wrapped_question}\nA: {answer}", fontsize=font_size*0.7, wrap=True)
        
        # Remove unused subplots if fewer than 5 images
        for col in range(num_images, images_per_class):
            ax = axes[row, col] if rows > 1 else axes[col]
            fig.delaxes(ax)
        

    # Adjust layout to add padding between rows
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Increase vertical space between subplots
    
    # Save the figure as a PDF
    plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free up memory
    
    print(f"Class-specific sample images saved as {path}")



def show_image(sample, predicted, target):
    plt.title(f'{sample["question"]}\nPredicted: {predicted}\nTarget: {target}')
    plt.imshow(sample["image"])
    plt.axis('off')
    plt.show()
    plt.close()