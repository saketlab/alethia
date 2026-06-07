import numpy as np


def do_pca(X: np.array, n_components=2, labels=None, return_expl_var=True):
    """Perform PCA dimensionality reduction."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    if return_expl_var:
        return X_pca, pca.explained_variance_ratio_ * 100
    return X_pca


def do_umap(X, n_components=2, random_state=42):
    """Perform UMAP dimensionality reduction."""
    import umap

    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)


def _palette_for(n_categories):
    """A qualitative palette with at least ``n_categories`` distinct colours."""
    # Set1 only has 9 colours; tab10/tab20 scale to the larger category counts that
    # real label sets (e.g. therapeutic classes) reach.
    if n_categories <= 9:
        return "Set1"
    if n_categories <= 10:
        return "tab10"
    if n_categories <= 20:
        return "tab20"
    return "husl"


def plot_embedding(
    X,
    labels=None,
    dims=[1, 2],
    color_map=None,
    title="",
    explained_var=None,
    axis_labels=None,
    label=False,
    repel=False,
    text_size=8,
    point_size=40,
):
    """Plot embeddings as a 2D scatter plot.

    Args:
        color_map: seaborn palette name. If ``None``, a qualitative palette sized to the
            number of distinct labels is chosen automatically.
        axis_labels: optional ``(xlabel, ylabel)`` (e.g. ``("UMAP 1", "UMAP 2")``). Falls
            back to PC labels with explained variance, or the column names.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X).astype(float)
        df.columns = [f"x{i}" for i in range(1, df.shape[1] + 1)]
    else:
        df = X.copy()
        df.columns = [f"x{i}" for i in range(1, df.shape[1] + 1)]

    if labels is not None and isinstance(labels, list):
        df["labels"] = labels

    if color_map is None and labels is not None:
        color_map = _palette_for(len(set(labels)))

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
        texts = labels if isinstance(labels, list) else [str(i) for i in range(len(df))]
        x_coords, y_coords = df[f"x{dims[0]}"].values, df[f"x{dims[1]}"].values

        if repel:
            try:
                from adjustText import adjust_text

                text_objs = [
                    plt.text(
                        x_coords[i], y_coords[i], str(texts[i]), fontsize=text_size
                    )
                    for i in range(len(texts))
                ]
                adjust_text(
                    text_objs, arrowprops=dict(arrowstyle="->", color="black", lw=0.5)
                )
            except ImportError:
                for i, txt in enumerate(texts):
                    plt.text(x_coords[i], y_coords[i], str(txt), fontsize=text_size)
        else:
            for i, txt in enumerate(texts):
                plt.text(x_coords[i], y_coords[i], str(txt), fontsize=text_size)

    plt.title(title)
    if axis_labels is not None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
    elif explained_var is not None:
        plt.xlabel(f"PC{dims[0]} ({explained_var[dims[0] - 1]:.2f}%)")
        plt.ylabel(f"PC{dims[1]} ({explained_var[dims[1] - 1]:.2f}%)")
    if labels is not None:
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def plot_embedding_df(
    df,
    x_col="x1",
    y_col="x2",
    label_by=None,
    color_by=None,
    color_map="Set1",
    title="",
    explained_var=None,
    label=False,
    repel=False,
    text_size=8,
    point_size=40,
):
    """Plot embeddings from DataFrame."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
    if label_by and label_by not in df.columns:
        raise ValueError(f"Column '{label_by}' not found")
    if color_by and color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found")

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=color_by,
        palette=color_map if color_by else None,
        s=point_size,
        alpha=1,
    )

    if label and label_by:
        texts = df[label_by].astype(str).tolist()
        x_coords, y_coords = df[x_col].values, df[y_col].values

        if repel:
            try:
                from adjustText import adjust_text

                text_objs = [
                    plt.text(x_coords[i], y_coords[i], texts[i], fontsize=text_size)
                    for i in range(len(texts))
                ]
                adjust_text(
                    text_objs, arrowprops=dict(arrowstyle="->", color="black", lw=0.5)
                )
            except ImportError:
                for i, txt in enumerate(texts):
                    plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)
        else:
            for i, txt in enumerate(texts):
                plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)

    plt.title(title)
    if explained_var is not None:
        plt.xlabel(f"PC1 ({explained_var[0]:.2f}%)")
        plt.ylabel(f"PC2 ({explained_var[1]:.2f}%)")
    else:
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    if color_by:
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
