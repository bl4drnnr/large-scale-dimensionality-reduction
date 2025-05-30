from typing import List, Literal

import plotly.express as px
import pandas as pd
import numpy as np
from plotly.graph_objs import Figure


def plot_reduced_embeddings(
    reduced_embeddings: np.ndarray, labels: List[str | int], method: str, type: Literal["2D", "3D"] = "2D"
) -> Figure:
    """
    Create a Plotly visualization of reduced embeddings with category labels.

    Parameters:
    reduced_embeddings : np.ndarray
        The reduced embeddings with shape (n_samples, 3).
    labels : List[str|int]
        The category labels corresponding to each embedding point.
    method : str
        The name of the dimensionality reduction method used (e.g., "UMAP", "t-SNE").
    type : Literal["2D", "3D"], default="2D"
        The type of visualization to create: "2D" for a 2D scatter plot or
        "3D" for a 3D scatter plot.

    Returns:
    Figure
        A Plotly figure object containing the visualization that can be displayed
        or saved.
    """

    df = pd.DataFrame(
        {
            "Dimension 1": reduced_embeddings[:, 0],
            "Dimension 2": reduced_embeddings[:, 1],
            "Dimension 3": reduced_embeddings[:, 2],
            "Label": labels,
        }
    )

    if type == "2D":
        fig = px.scatter(
            df,
            x="Dimension 1",
            y="Dimension 2",
            color="Label",
            title=f"{method} Projection",
            labels={"Label": "Categories"},
            opacity=0.7,
            height=600,
            width=800,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig.update_traces(marker=dict(size=8))

    elif type == "3D":
        fig = px.scatter_3d(
            df,
            x="Dimension 1",
            y="Dimension 2",
            z="Dimension 3",
            color="Label",
            title=f"{method} Projection",
            labels={"Label": "Categories"},
            opacity=0.7,
            height=600,
            width=800,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig.update_traces(marker=dict(size=5))

    fig.update_layout(
        legend_title_text="Categories", legend=dict(itemsizing="constant"), margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig
