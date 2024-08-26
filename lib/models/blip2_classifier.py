import logging
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import StandardScaler
from transformers import Blip2Config
from umap import UMAP
import numpy as np
import wandb
from lib.types import DatasetTypes
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import plotly.graph_objects as go
import plotly.io as pio
from .base_classifier import Blip2, Blip2ClassifierConfig
import umap
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)


class Blip2Classifier(Blip2):
    config_class = Blip2ClassifierConfig

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # Define dimensions for the combined features
        combined_dim = config.vision_config.hidden_size + config.qformer_config.hidden_size

        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space = config.answer_space
        self.id_to_answer = {idx: answer for idx, answer in enumerate(self.answer_space)}
        
        self.umap_model = UMAP(n_neighbors=30, metric="euclidean", min_dist=0.01, n_components=2, random_state=42)

        # TODO[RF] different networks
        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, 1536),  # 2176 -> 1536
                nn.ReLU(),
                nn.BatchNorm1d(1536),
                nn.Dropout(0.4),
                nn.Linear(1536, 1024),  # 1536 -> 1024
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),  # 1024 -> 512
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, len(self.answer_space)),  # 512 -> 12
            )
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, 1024),  # Reduced layer count: 2176 -> 1024 directly
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, len(self.answer_space)),
            )

        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError()

        self.classifier.apply(self.initialize_weights)

        self.accumulated_features = []
        self.accumulated_labels = []

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def visualize_features_with_umap(self, interactive = True, save_format='html', sample_size=5000, aggregate=True):
        # Convert lists to tensors and move to CPU
        features = torch.cat(self.accumulated_features, dim=0).detach().cpu().numpy()
        labels_tensor = torch.cat(self.accumulated_labels, dim=0).detach().cpu().numpy()
            
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        if aggregate:    
            labels = np.argmax(labels_tensor, axis=1)
            unique_labels = np.unique(labels)

            scaled_features = np.array([scaled_features[labels == label].mean(axis=0) for label in unique_labels])
            labels = unique_labels
        else:
            # Stratified sampling for better visibility
            unique_labels, counts = np.unique(np.argmax(labels_tensor, axis=1), return_counts=True)
            samples_per_class = sample_size // len(unique_labels)  # Calculate samples per class

            sampled_indices = []
            for label in unique_labels:
                class_indices = np.where(np.argmax(labels_tensor, axis=1) == label)[0]
                if len(class_indices) > samples_per_class:
                    sampled_indices.extend(np.random.choice(class_indices, samples_per_class, replace=False))
                else:
                    sampled_indices.extend(class_indices)

            sampled_indices = np.array(sampled_indices)

            scaled_features = scaled_features[sampled_indices]
            labels = labels_tensor[sampled_indices]

        # Apply UMAP
        umap_embedding = self.umap_model.fit_transform(scaled_features)

        class_names = np.array([self.id_to_answer[label] for label in labels])
        
        # Prepare labels for color mapping
        unique_labels = np.unique(np.argmax(labels_tensor, axis=1))
        label_to_color = {label: i for i, label in enumerate(unique_labels)}

        # Map labels to colors
        colors = [label_to_color[label] for label in unique_labels]

        if interactive:
            # Create interactive Plotly figure
            fig = go.Figure(data=go.Scattergl(
                x=umap_embedding[:, 0],
                y=umap_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=colors,
                    colorscale='Viridis', 
                    size=10,
                    opacity=0.8,
                    line=dict(width=1, color='black'), 
                ),
                text=class_names,
                hoverinfo='text+x+y'
            ))

            fig.update_layout(
                title="UMAP projection of aggregated features",
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                legend_title="Classes",
                width=800,
                height=600
            )

            # Save as interactive HTML or JSON based on specified format
            if save_format == 'html':
                pio.write_html(fig, file='umap_features_interactive.html')
            elif save_format == 'json':
                pio.write_json(fig, file='umap_features_interactive.json')

        else:
            # Optional: Create a static plot for quick viewing
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                umap_embedding[:, 0], 
                umap_embedding[:, 1], 
                c=colors, 
                cmap='viridis', 
                alpha=0.8,  # Slightly increase opacity
                s=40,  # Increase size for better visibility
                edgecolors='black'  # Add edge color to markers
            )
            
            # Create a color bar with actual class names
            cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_labels)), label='Class Labels')
            
            # Update colorbar labels to actual class names
            cbar.set_ticks(np.arange(len(unique_labels)))
            cbar.set_ticklabels([self.id_to_answer[label] for label in unique_labels])
            
            plt.title("UMAP projection of aggregated features (Static)")
            plt.savefig("umap_features_static.svg")
            plt.close()
                    
    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        extract_features: bool = False,
    ):
        # Extract image features
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        # Get vision features
        vision_pooled = outputs.vision_outputs.last_hidden_state.mean(dim=1)

        # Get Q-Former features and pool them
        qformer_pooled = outputs.qformer_outputs["last_hidden_state"].mean(dim=1)

        # Combine vision and Q-Former features
        combined_features = torch.cat((vision_pooled, qformer_pooled), dim=1)

        # Classification
        logits = self.classifier(combined_features)

        if extract_features:
            # Accumulate features and labels
            self.accumulated_features.append(combined_features)
            if labels is not None:
                self.accumulated_labels.append(labels)
        else:
            self.visualize_features_with_umap()

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            combined_loss = outputs.loss + classifier_loss
            outputs.loss = combined_loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {classifier_loss}")
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs

    def visualize_with_umap_and_density(self, save_path='umap_density_plot.html'):
        # Assuming features is your (8000, 2176) tensor
        features = torch.cat(self.accumulated_features, dim=0).detach().cpu().numpy()
        labels_tensor = torch.cat(self.accumulated_labels, dim=0).detach().cpu().numpy()
        
        # Apply UMAP
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_embedding = umap_model.fit_transform(features)

        # Create a DataFrame for Plotly
        df = pd.DataFrame(umap_embedding, columns=['UMAP1', 'UMAP2'])
        df['Class'] = np.argmax(labels_tensor, axis=1)  # Assuming labels are one-hot encoded

        # Create scatter plot
        fig = px.scatter(df, x='UMAP1', y='UMAP2', color='Class', title='UMAP Projection of Features with Density',
                         labels={'Class': 'Class Labels'}, opacity=0.7)

        # Add density contours for each class
        for class_label in df['Class'].unique():
            class_data = df[df['Class'] == class_label]
            x = class_data['UMAP1']
            y = class_data['UMAP2']
            
            # Perform kernel density estimation
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
            # Create a grid of points
            xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            
            # Add contour plot
            fig.add_trace(go.Contour(
                x=xi.flatten(),
                y=yi.flatten(),
                z=zi,
                colorscale='Viridis',
                showscale=False,
                opacity=0.3,
                name=f'Density Class {class_label}',
                contours=dict(showlabels=False),
                hoverinfo='skip'
            ))

        # Update layout for better visibility
        fig.update_layout(
            legend_title_text='Classes',
            width=900,
            height=700
        )

        # Display the plot
        fig.show()

        # Save the plot as an interactive HTML file
        if save_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")


class Blip2Classifier2(Blip2):
    config_class = Blip2ClassifierConfig

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # 1408 + 2560
        # Fusion and final classification
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space_dim = config.answer_space

        # Layer Normalization
        self.qformer_norm = nn.LayerNorm(config.qformer_config.hidden_size)  # 768
        self.vit_norm = nn.LayerNorm(config.vision_config.hidden_size)  # 1408

        self.qformer_proj = nn.Linear(config.qformer_config.hidden_size, 512)
        self.vit_proj = nn.Linear(config.vision_config.hidden_size, 1024)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 1024, config.interm_dim),  # 32 x 768
            # nn.BatchNorm1d(config.interm_dim),  # BatchNorm before ReLU
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, self.answer_space_dim),
        )

        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError()

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Extract image features
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )
        vit_features = outputs.vision_outputs.last_hidden_state.mean(dim=1)
        qformer_features = outputs.qformer_outputs["last_hidden_state"].mean(dim=1)
        # Normalize features
        qformer_features = self.qformer_norm(qformer_features)
        vit_features = self.vit_norm(vit_features)

        # Project features to same dimension
        qformer_features = self.qformer_proj(qformer_features)
        vit_features = self.vit_proj(vit_features)

        # concatenate features
        combined_features = torch.cat((qformer_features, vit_features), dim=-1)

        # Classification
        logits = self.classifier(combined_features)

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            combined_loss = outputs.loss + classifier_loss
            outputs.loss = combined_loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {classifier_loss}")
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs


class Blip2Classifier3(Blip2):
    config_class = Blip2ClassifierConfig

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # 1408 + 2560
        # Fusion and final classification
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space_dim = config.answer_space

        self.classifier = nn.Sequential(
            nn.Linear(config.classification_input_dim, config.interm_dim),  # 32 x 768
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, self.answer_space_dim),
        )

        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError()

        # self.classifier.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Extract image features
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        features = outputs.qformer_outputs["pooler_output"]

        # Classification
        logits = self.classifier(features)

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            combined_loss = outputs.loss + classifier_loss
            outputs.loss = combined_loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {classifier_loss}")
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs
