import logging
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

class FeatureVisualizer:
    """
    This class is used to visualize the features. It uses UMAP to project the features into 
    a 2D space.
    """
    def __init__(self, id_to_answer):
        self.id_to_answer = id_to_answer
        self.accumulated_features = []
        self.accumulated_labels = []
        self.umap_model = UMAP(
            n_neighbors=30,
            metric="euclidean",
            min_dist=0.01,
            n_components=2,
            random_state=42,
        )

    def accumulate_features(self, features, labels):
        self.accumulated_features.append(features)
        if labels is not None:
            self.accumulated_labels.append(labels)

    def visualize_features_with_umap(
        self, interactive=True, save_format="html", sample_size=5000, aggregate=True
    ):
        features, labels = self._prepare_data()
        scaled_features = self._scale_features(features)
        
        if aggregate:
            scaled_features, labels = self._aggregate_features(scaled_features, labels)
        else:
            scaled_features, labels = self._sample_features(scaled_features, labels, sample_size)

        umap_embedding = self.umap_model.fit_transform(scaled_features)
        class_names = np.array([self.id_to_answer[label] for label in labels])
        colors = self._prepare_colors(labels)

        if interactive:
            self._create_interactive_plot(umap_embedding, colors, class_names, save_format)
        else:
            self._create_static_plot(umap_embedding, colors, labels)

    def _prepare_data(self):
        features = torch.cat(self.accumulated_features, dim=0).detach().cpu().numpy()
        labels = torch.cat(self.accumulated_labels, dim=0).detach().cpu().numpy()
        return features, labels

    def _scale_features(self, features):
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def _aggregate_features(self, scaled_features, labels):
        labels = np.argmax(labels, axis=1)
        unique_labels = np.unique(labels)
        aggregated_features = np.array([
            scaled_features[labels == label].mean(axis=0)
            for label in unique_labels
        ])
        return aggregated_features, unique_labels

    def _sample_features(self, scaled_features, labels, sample_size):
        unique_labels, counts = np.unique(np.argmax(labels, axis=1), return_counts=True)
        samples_per_class = sample_size // len(unique_labels)

        sampled_indices = []
        for label in unique_labels:
            class_indices = np.where(np.argmax(labels, axis=1) == label)[0]
            if len(class_indices) > samples_per_class:
                sampled_indices.extend(np.random.choice(class_indices, samples_per_class, replace=False))
            else:
                sampled_indices.extend(class_indices)

        sampled_indices = np.array(sampled_indices)
        return scaled_features[sampled_indices], labels[sampled_indices]

    def _prepare_colors(self, labels):
        unique_labels = np.unique(np.argmax(labels, axis=1))
        label_to_color = {label: i for i, label in enumerate(unique_labels)}
        return [label_to_color[label] for label in unique_labels]

    def _create_interactive_plot(self, umap_embedding, colors, class_names, save_format):
        fig = go.Figure(data=go.Scattergl(
            x=umap_embedding[:, 0],
            y=umap_embedding[:, 1],
            mode="markers",
            marker=dict(
                color=colors,
                colorscale="Viridis",
                size=10,
                opacity=0.8,
                line=dict(width=1, color="black"),
            ),
            text=class_names,
            hoverinfo="text+x+y",
        ))

        fig.update_layout(
            title="UMAP projection of aggregated features",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Classes",
            width=800,
            height=600,
        )

        if save_format == "html":
            pio.write_html(fig, file="umap_features_interactive.html")
        elif save_format == "json":
            pio.write_json(fig, file="umap_features_interactive.json")

    def _create_static_plot(self, umap_embedding, colors, labels):
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
        cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_labels)), label="Class Labels")
        cbar.set_ticks(np.arange(len(unique_labels)))
        cbar.set_ticklabels([self.id_to_answer[label] for label in unique_labels])

        plt.title("UMAP projection of aggregated features (Static)")
        plt.savefig("umap_features_static.svg")
        plt.close()

    def visualize_with_umap_and_density(self, save_path="umap_density_plot.html"):
        features = torch.cat(self.accumulated_features, dim=0).detach().cpu().numpy()
        labels_tensor = torch.cat(self.accumulated_labels, dim=0).detach().cpu().numpy()

        umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_embedding = umap_model.fit_transform(features)

        df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
        df["Class"] = np.argmax(labels_tensor, axis=1)

        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color="Class",
            title="UMAP Projection of Features with Density",
            labels={"Class": "Class Labels"},
            opacity=0.7,
        )

        for class_label in df["Class"].unique():
            class_data = df[df["Class"] == class_label]
            x, y = class_data["UMAP1"], class_data["UMAP2"]
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

            fig.add_trace(
                go.Contour(
                    x=xi.flatten(),
                    y=yi.flatten(),
                    z=zi,
                    colorscale="Viridis",
                    showscale=False,
                    opacity=0.3,
                    name=f"Density Class {class_label}",
                    contours=dict(showlabels=False),
                    hoverinfo="skip",
                )
            )

        fig.update_layout(legend_title_text="Classes", width=900, height=700)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")

        return fig