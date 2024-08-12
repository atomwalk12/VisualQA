# @title Setup paths

# Load the embeddings

import logging
import time

import numpy as np
import plotly.express as px
import torch
import umap
from colorama import Fore
from tqdm import tqdm

from lib.daquar.daquar_visualization import \
    VisualizationDaquarGeneration
from lib.easy_vqa.easyvqa_visualization import \
    VisualizationEasyVQAGeneration
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.generation_trainer import GenerationTrainer
from lib.types import TrainingParameters
from lib.utils import format_time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lib.types import SAVE_PATHS, DatasetTypes, FileNames, ModelTypes, State
from ..types import DatasetTypes, FileNames
from .base_trainer import TorchBase
from ..daquar.daquar_base import DaquarDatasetBase
from ..easy_vqa.easyvqa_base import EasyVQADatasetBase

logger = logging.getLogger(__name__)


class VisualizeGenerator(GenerationTrainer):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)
        self.blue = Fore.BLUE

    def generate_embeddings(self):
        """
        Create embeddings to visualize and measure similarities between images.
        The Image Transformer outputs' embedding dimension size is (batch_size, num_query_tokens=32, embedding_dim=768)
        However, we use the pooled results which are of dimension (batch_size, embedding_dim=768)
        """
        # Initialize the statistics
        start = time.time()

        # Generate embeddings
        self.model.eval()

        bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for step, (data, indices) in bar:
            # Unpack the batch
            input_ids, pixel_values, attention_mask, labels = self.send_to_device_if_needed(data)

            # Generate the output
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,
                attention_mask=attention_mask,
            )
            self.state.history["items"].append(
                [self.train_dataloader.dataset.raw_dataset[idx.item()]["image"] for idx in indices]
            )

            self.update_state_with_embeddings(outputs)

        # Now report the results
        end = time.time()
        time_elapsed = end - start
        print(f"{self.blue}Generating embeddings completed in {format_time(time_elapsed)}")
        print()

        self.state.save_state_to_file(self.best_path, file_name=FileNames.UMAPEmbedding.format(self.config.split))

        return self.state

    def display(self, history, filename, num_samples=80):
        # umap reducer
        data = torch.concatenate([embedding["pooler_output"] for embedding in history.history["embeddings"]], axis=0)
        items = [item for sublist in history.history["items"] for item in sublist]
        reducer = umap.UMAP()
        embeddings = reducer.fit_transform(data)

        # pick up randomly
        rand_index = np.random.choice(len(embeddings), num_samples, replace=False)
        rand_embed = embeddings[rand_index]
        rand_items = [items[i] for i in rand_index]

        fig = px.scatter(x=rand_embed[:, 0], y=rand_embed[:, 1])

        for d, item in zip(rand_embed, rand_items):
            fig.add_layout_image(
                x=d[0],
                y=d[1],
                source=item,
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                sizex=0.5,
                sizey=0.5,
            )

        fig.write_image(f"{self.best_path}/{filename}")
        return fig

    def get_image_path(self, data):
        data["item"] = self.train_dataloader.dataset

    def get_dataset(self, args):
        if self.dataset_name == DatasetTypes.EASY_VQA:
            return VisualizationEasyVQAGeneration(args)
        else:
            return VisualizationDaquarGeneration(args)

class VisualizeClassifier(ClassificationTrainer):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)
        if self.dataset_name == DatasetTypes.DAQUAR:
            self.answer_space = DaquarDatasetBase.get_answers()
        elif self.dataset_name == DatasetTypes.EASY_VQA:
            self.answer_space = EasyVQADatasetBase.get_answer()

    def confusion_matrix(self, matrix: State):
        all_preds = matrix.history['confusion_predictions']
        all_labels = matrix.history['confusion_labels']
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.answer_space)))

        # Plot the confusion matrix
        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=self.answer_space, yticklabels=self.answer_space)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        fig.savefig(f"{self.best_path}/{FileNames.ConfusionMatrixPDF.format(self.config.split)}")
        
    
    def load_confusion_matrix(self, split):
        state = State()
        return state.load_state(self.best_path, FileNames.ConfusionMatrix.format(split))