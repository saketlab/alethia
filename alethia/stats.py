import numpy as np


def do_pca(X: np.array, n_components=2, labels=None, return_expl_var=True):
    """Performs Principal Component Analysis (PCA) on the input data.

    Reduces the dimensionality of the input data X to 2 principal components
    using scikit-learn's PCA implementation. Optionally returns the explained
    variance ratio for each component.

    Args:
        X (np.array): The input data as a NumPy array.
        labels: Labels for the data points (optional). Not used in PCA calculation.
        return_expl_var (bool, optional): Whether to return the explained variance. Defaults to True.

    Returns:
        tuple or np.array: If return_expl_var is True, returns a tuple containing the
                           transformed data and the explained variance ratio. Otherwise,
                           returns only the transformed data.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_ * 100
    if return_expl_var:
        return X, explained_var
    return X


def do_umap(X, n_components=2, random_state=42):
    """Performs dimensionality reduction using UMAP.

    Reduces the dimensionality of the input data X using the UMAP algorithm.

    Args:
        X (np.array): The input data as a NumPy array.
        n_components (int, optional): The number of dimensions to reduce to. Defaults to 2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        np.array: The transformed data with reduced dimensionality.
    """
    import umap

    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    X_umap = reducer.fit_transform(X)
    return X_umap


def plot_embedding(
    X,
    labels=None,
    dims=[1, 2],
    color_map="Set1",
    title="",
    explained_var=None,
    label=False,
    repel=False,
    text_size=8,
    point_size=40,
):
    """Plots embeddings in a 2D scatter plot with optional text labels that can be repelled.

    Args:
        X (np.ndarray or pd.DataFrame): The embedding data. Can be a NumPy array or a Pandas DataFrame.
                                        If a NumPy array, it should be 2D. If a DataFrame, x_col and y_col
                                        should be column names.
        labels (list or str, optional): Labels for coloring the points. Can be a list or a string
                                         (DataFrame column name). Defaults to None.
        dims (list, optional): Dimensions to plot. Defaults to [1, 2].
        color_map (str, optional): Seaborn color map name. Defaults to "Set1".
        title (str, optional): Plot title. Defaults to "".
        explained_var (list, optional): Explained variance for each dimension, used for PCA plots. Defaults to None.
        label (bool, optional): Whether to add text labels to the points. Defaults to False.
        repel (bool, optional): Whether to repel text labels to avoid overlap. Requires label=True. Defaults to False.
        text_size (int, optional): Font size for text labels. Defaults to 8.
        point_size (int, optional): Size of scatter points. Defaults to 40.

    Raises:
        TypeError: If input `X` is neither a NumPy array nor a Pandas DataFrame.
        ValueError: If `explained_var` is provided but not a list or of incorrect length.
        ImportError: If repel=True but adjustText is not installed.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X).astype(float)
        df.columns = [f"x{i}" for i in range(1, df.shape[1] + 1)]
        if labels is not None:
            df["labels"] = labels
    elif isinstance(X, pd.DataFrame):
        df = X.copy()  # Avoid modifying the original DataFrame
        df.columns = [f"x{i}" for i in range(1, df.shape[1] + 1)]
        if labels is not None and isinstance(labels, list):
            df["labels"] = labels
    else:
        raise TypeError("X must be a NumPy array or a Pandas DataFrame.")

    scatter = sns.scatterplot(
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
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                    expand_points=(1.5, 1.5),
                    force_points=(0.1, 0.1),
                )
            except ImportError:
                print("Warning: The 'adjustText' library is required for repel=True.")
                print("Install it using: pip install adjustText")

                # Fall back to regular text labels without repelling
                for i, txt in enumerate(texts):
                    plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)
        else:
            for i, txt in enumerate(texts):
                plt.text(x_coords[i], y_coords[i], txt, fontsize=text_size)

    plt.title(title)

    # Add explained variance info if provided
    if explained_var is not None:
        if not isinstance(explained_var, (list, np.ndarray)) or len(explained_var) < 2:
            raise ValueError("explained_var must be a list with at least two values.")
        plt.xlabel(f"PC{dims[0]} ({explained_var[dims[0]-1]:.2f}%)")
        plt.ylabel(f"PC{dims[1]} ({explained_var[dims[1]-1]:.2f}%)")

    # Add legend if labels provided
    if labels is not None:
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.show()
