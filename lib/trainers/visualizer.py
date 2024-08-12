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
from lib.trainers.generation_trainer import GenerationTrainer
from lib.types import TrainingParameters
from lib.utils import format_time

from ..types import DatasetTypes

logger = logging.getLogger(__name__)


class Visualizer(GenerationTrainer):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)
        self.save_embeddings = True
        self.file_name = "embeddings.pkl"
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

        self.state.save_embeddings(self.best_path, file_name=self.file_name)

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
