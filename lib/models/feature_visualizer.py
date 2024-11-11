import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import colorsys
from lib.types import SAVE_PATHS

logger = logging.getLogger(__name__)


class FeatureVisualizer:
    """
    This class is used to visualize the features. It uses UMAP to project the features into
    a 2D space.
    """

    def __init__(self, id_to_answer, dataset_name):
        self.id_to_answer = id_to_answer
        self.accumulated_features_train = []
        self.accumulated_labels_train = []
        self.accumulated_features_valid = []
        self.accumulated_labels_valid = []
        self.umap_model = UMAP(
            n_neighbors=13,
            metric="euclidean",
            min_dist=0.01,
            n_components=2,
            random_state=42,
        )
        self.dataset_name = dataset_name
        self.best_path = (
            SAVE_PATHS.BLIP2_Classifier_DAQUAR
            if dataset_name == "daquar"
            else SAVE_PATHS.BLIP2_Classifier_EasyVQA
        )

    def accumulate_features(self, features, labels, split):
        if split == "train":
            self.accumulated_features_train.append(features)
            self.accumulated_labels_train.append(labels)
        elif split == "val":
            self.accumulated_features_valid.append(features)
            self.accumulated_labels_valid.append(labels)

    def get_features(self, split):
        if split == "train":
            features = {
                "features": self.accumulated_features_train,
                "labels": self.accumulated_labels_train,
            }
        elif split == "val":
            features = {
                "features": self.accumulated_features_valid,
                "labels": self.accumulated_labels_valid,
            }
        return features

    def set_features(self, features, labels, split):
        if split == "train":
            self.accumulated_features_train = features
            self.accumulated_labels_train = labels
        elif split == "val":
            self.accumulated_features_valid = features
            self.accumulated_labels_valid = labels
        self.split = split

    def reset(self, epoch, is_better):
        print(
            f"Saving feature map to {self.best_path}/images/{is_better}_{self.dataset_name}_{epoch}_features"
        )
        self.split = "val"
        self.visualize_features_with_umap(
            f"{self.best_path}/images/{is_better}_{self.dataset_name}_{epoch}_features",
            show=False,
        )
        self.accumulated_features_train = []
        self.accumulated_labels_train = []
        self.accumulated_features_valid = []
        self.accumulated_labels_valid = []

    def visualize_features_with_umap(
        self,
        save_path,
        interactive=True,
        save_format="html",
        sample_size=5000,
        aggregate=False,
        show=True,
    ):
        """
        Visualize the features that are aggregated by the classifier.
        """
        features, labels = self._prepare_data()
        scaled_features = self._scale_features(features)

        if aggregate:
            scaled_features, labels = self._aggregate_features(scaled_features, labels)
        else:
            scaled_features, labels = self._sample_features(
                scaled_features, labels, sample_size
            )

        umap_embedding = self.umap_model.fit_transform(scaled_features)

        # Convert one-hot encoded labels to class indices
        label_indices = np.argmax(labels, axis=1)

        # Get class names using the indices
        class_names = np.array([self.id_to_answer[idx] for idx in label_indices])

        colors = self._prepare_colors(label_indices)

        if interactive:
            self._create_interactive_plot(
                umap_embedding, colors, class_names, save_format, save_path, show
            )
        else:
            self._create_static_plot(umap_embedding, colors, label_indices, save_path)

    def _prepare_data(self):
        if self.split == "train":
            features = (
                torch.cat(self.accumulated_features_train, dim=0).detach().cpu().numpy()
            )
            labels = (
                torch.cat(self.accumulated_labels_train, dim=0).detach().cpu().numpy()
            )
        elif self.split == "val":
            features = (
                torch.cat(self.accumulated_features_valid, dim=0).detach().cpu().numpy()
            )
            labels = (
                torch.cat(self.accumulated_labels_valid, dim=0).detach().cpu().numpy()
            )
        return features, labels

    def _scale_features(self, features):
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def _aggregate_features(self, scaled_features, labels):
        labels = np.argmax(labels, axis=1)
        unique_labels = np.unique(labels)
        aggregated_features = np.array(
            [scaled_features[labels == label].mean(axis=0) for label in unique_labels]
        )
        return aggregated_features, unique_labels

    def _sample_features(self, scaled_features, labels, sample_size):
        num_classes = labels.shape[1]
        samples_per_class = sample_size // num_classes

        sampled_indices = []
        for class_idx in range(num_classes):
            class_indices = np.where(labels[:, class_idx] == 1)[0]
            if len(class_indices) > samples_per_class:
                sampled_indices.extend(
                    np.random.choice(class_indices, samples_per_class, replace=False)
                )
            else:
                sampled_indices.extend(class_indices)

        sampled_indices = np.array(sampled_indices)
        return scaled_features[sampled_indices], labels[sampled_indices]

    def _prepare_colors(self, label_indices):
        unique_labels = np.unique(label_indices)
        num_colors = len(unique_labels)

        # Start with Plotly's qualitative colors
        base_colors = px.colors.qualitative.Plotly

        # Generate additional colors if needed
        if num_colors > len(base_colors):
            additional_colors = self._generate_colors(num_colors - len(base_colors))
            color_map = base_colors + additional_colors
        else:
            color_map = base_colors[:num_colors]

        label_to_color = {
            label: color for label, color in zip(unique_labels, color_map)
        }
        return [label_to_color[label] for label in label_indices]

    def _generate_colors(self, n):
        HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        return [
            "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in RGB_tuples
        ]

    def _create_interactive_plot(
        self, umap_embedding, colors, class_names, save_format, save_path, show=True
    ):
        unique_classes = np.unique(class_names)
        traces = []

        for class_name in unique_classes:
            mask = class_names == class_name
            traces.append(
                go.Scattergl(
                    x=umap_embedding[mask, 0],
                    y=umap_embedding[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=5,
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=class_names[mask],
                    hoverinfo="text+x+y",
                    name=class_name,
                    showlegend=True,
                )
            )

        fig = go.Figure(data=traces)

        fig.update_layout(
            title=dict(
                text=f"UMAP feature projections ({self.split})",
                font=dict(size=24),  # Increased title font size
            ),
            xaxis_title=dict(
                text="UMAP1",
                font=dict(size=18),  # Increased axis title font size
            ),
            yaxis_title=dict(text="UMAP2", font=dict(size=18)),
            legend_title=dict(text="Classes", font=dict(size=18)),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                font=dict(size=14),  # Increased legend font size
            ),
            width=1000,
            height=800,
            hovermode="closest",
        )

        # Increase axis label font sizes
        fig.update_xaxes(showgrid=False, tickfont=dict(size=14))
        fig.update_yaxes(showgrid=False, tickfont=dict(size=14))

        if show:
            if save_format == "html":
                pio.write_html(fig, file=f"{save_path}.html")
                pio.write_image(fig, file=f"{save_path}.pdf", format="pdf")
                pio.write_image(fig, file=f"{save_path}.svg", format="svg")
                pio.write_image(fig, file=f"{save_path}.jpg", format="jpg")
            elif save_format == "json":
                pio.write_json(fig, file=f"{save_path}.json")
                pio.write_image(fig, file=f"{save_path}.pdf", format="pdf")
                pio.write_image(fig, file=f"{save_path}.svg", format="svg")
            fig.show()
        else:
            pio.write_image(fig, file=f"{save_path}.jpg", format="jpg")

        return fig

    def _create_static_plot(self, umap_embedding, colors, labels, save_path):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=colors,
            cmap="viridis",
            alpha=0.8,
            s=40,
            edgecolors="black",
        )

        unique_labels = np.unique(labels)
        cbar = plt.colorbar(
            scatter, ticks=np.arange(len(unique_labels)), label="Class Labels"
        )
        cbar.set_ticks(np.arange(len(unique_labels)))
        cbar.set_ticklabels([self.id_to_answer[label] for label in unique_labels])

        plt.title("UMAP projection of aggregated features (Static)")
        plt.savefig(f"{save_path}.svg")
        plt.close()
