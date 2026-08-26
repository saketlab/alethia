from collections.abc import Sequence
from typing import Any

import numpy as np


def do_pca(
    X: np.ndarray, n_components=2, labels=None, return_expl_var=True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Reduce ``X`` to 2 principal components."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_ * 100
    if return_expl_var:
        return X_pca, explained_var
    return X_pca


def do_umap(X, n_components=2, random_state=42) -> np.ndarray:
    """Reduce ``X`` to ``n_components`` dimensions with UMAP."""
    import umap

    return umap.UMAP(
        n_components=n_components, random_state=random_state
    ).fit_transform(X)


def plot_embedding(
    X: Any,
    labels: Any = None,
    dims=(1, 2),
    color_map="Set1",
    title="",
    explained_var: Sequence[float] | None = None,
    label=False,
    repel: bool = False,
    text_size=8,
    point_size=40,
):
    """Scatter-plot a 2D embedding, optionally with repelled text labels."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X).astype(float)
        df.columns = [f"x{i}" for i in range(1, df.shape[1] + 1)]
        if labels is not None:
            df["labels"] = labels
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
        df.columns = [f"x{i}" for i in range(1, df.shape[1] + 1)]
        if labels is not None and isinstance(labels, list):
            df["labels"] = labels
    else:
        raise TypeError("X must be a NumPy array or a Pandas DataFrame.")

    sns.scatterplot(
        data=df,
        x=f"x{dims[0]}",
        y=f"x{dims[1]}",
        hue=labels if labels is not None else None,
        palette=color_map if labels is not None else None,
        s=point_size,
        alpha=1,
    )

    if label:
        if isinstance(labels, str) and labels in df.columns:
            texts = df[labels].astype(str).tolist()
        elif isinstance(labels, list):
            texts = [str(label) for label in labels]
        else:
            texts = [str(i) for i in range(len(df))]

        x_coords = df[f"x{dims[0]}"].values
        y_coords = df[f"x{dims[1]}"].values

        if repel:
            try:
                from adjustText import adjust_text

                text_objects = []
                for i, txt in enumerate(texts):
                    text_objects.append(
                        plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)
                    )

                adjust_text(
                    text_objects,
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.5},
                    expand_points=(1.5, 1.5),
                    force_points=(0.1, 0.1),
                )
            except ImportError:
                print("Warning: The 'adjustText' library is required for repel=True.")
                print("Install it using: pip install adjustText")

                for i, txt in enumerate(texts):
                    plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)
        else:
            for i, txt in enumerate(texts):
                plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)

    plt.title(title)

    if explained_var is not None:
        if not isinstance(explained_var, (list, np.ndarray)) or len(explained_var) < 2:
            raise ValueError("explained_var must be a list with at least two values.")
        plt.xlabel(f"PC{dims[0]} ({explained_var[dims[0] - 1]:.2f}%)")
        plt.ylabel(f"PC{dims[1]} ({explained_var[dims[1] - 1]:.2f}%)")

    if labels is not None:
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.show()


def plot_embedding_df(
    df,
    x_col="x1",
    y_col="x2",
    label_by: str | None = None,
    color_by: str | None = None,
    color_map="Set1",
    title="",
    explained_var: Sequence[float] | None = None,
    label=False,
    repel: bool = False,
    text_size=8,
    point_size=40,
):
    """Scatter-plot embeddings from a DataFrame, colouring and labelling by separate columns."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame")
    if label_by is not None and label_by not in df.columns:
        raise ValueError(f"Column '{label_by}' not found in DataFrame")
    if color_by is not None and color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found in DataFrame")

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=color_by if color_by is not None else None,
        palette=color_map if color_by is not None else None,
        s=point_size,
        alpha=1,
    )

    if label and label_by is not None:
        texts = df[label_by].astype(str).tolist()
        x_coords = df[x_col].values
        y_coords = df[y_col].values

        if repel:
            try:
                from adjustText import adjust_text

                text_objects = []
                for i, txt in enumerate(texts):
                    text_objects.append(
                        plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)
                    )

                adjust_text(
                    text_objects,
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.5},
                    expand_points=(1.5, 1.5),
                    force_points=(0.1, 0.1),
                )
            except ImportError:
                print("Warning: The 'adjustText' library is required for repel=True.")
                print("Install it using: pip install adjustText")

                for i, txt in enumerate(texts):
                    plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)
        else:
            for i, txt in enumerate(texts):
                plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)

    plt.title(title)

    if explained_var is not None:
        if not isinstance(explained_var, (list, np.ndarray)) or len(explained_var) < 2:
            raise ValueError("explained_var must be a list with at least two values.")
        plt.xlabel(f"PC1 ({explained_var[0]:.2f}%)")
        plt.ylabel(f"PC2 ({explained_var[1]:.2f}%)")
    else:
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    if color_by is not None:
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.show()
