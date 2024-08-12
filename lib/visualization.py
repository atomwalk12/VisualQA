from typing import List
from PIL.Image import Image
import matplotlib.pyplot as plt
import numpy as np

from lib.trainers.visualizer import VisualizeGenerator, VisualizeClassifier
from lib.types import SAVE_PATHS, DatasetTypes, FileNames, ModelTypes, State, Suffix, TrainingParameters, VQAParameters


def show_images_with_captions(images_or_paths: List[Image] | List[str], captions: List[str], cols=3):
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
    assert split.startswith(Suffix.Test) or split.startswith(Suffix.Val)
    
    # Prepare the parameters
    train_args = VQAParameters(split=split, use_stratified_split=True)
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

    visualizer.display(history, num_samples=num_samples, filename=FileNames.UMAPClustering.format(split))


def show_confusion_matrix(split, dataset):  
    args = VQAParameters(split=split, use_stratified_split=True)
      
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
        split=split
    )
    
    visualizer = VisualizeClassifier(parameters)
    
    confusion_matrix = visualizer.load_confusion_matrix(split)
    visualizer.confusion_matrix(confusion_matrix)
