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
    X, labels=None, dims=[1, 2], color_map="Set1", title="", explained_var=None
):
    """Plots embeddings in a 2D scatter plot.

    Args:
        X (np.ndarray or pd.DataFrame): The embedding data.  Can be a NumPy array or a Pandas DataFrame.
                                        If a NumPy array, it should be 2D.  If a DataFrame, x_col and y_col
                                        should be column names.
        labels (list or str, optional): Labels for coloring the points. Can be a list or a string
                                         (DataFrame column name). Defaults to None.
        dims (list, optional): Dimensions to plot. Defaults to [1, 2].
        color_map (str, optional): Seaborn color map name. Defaults to "Set1".
        title (str, optional): Plot title. Defaults to "".
        explained_var (list, optional): Explained variance for each dimension, used for PCA plots. Defaults to None.

    Raises:
        TypeError: If input `X` is neither a NumPy array nor a Pandas DataFrame.
        ValueError: If `explained_var` is provided but not a list or of incorrect length.

    """
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
        s=40,
        alpha=1,
    )
    plt.title(title)

    if explained_var is not None:
        if not isinstance(explained_var, (list, np.ndarray)) or len(explained_var) < 2:
            raise ValueError("explained_var must be a list with at least two values.")
        plt.xlabel(f"PC{dims[0]} ({explained_var[dims[0]-1]:.2f}%)")
        plt.ylabel(f"PC{dims[1]} ({explained_var[dims[1]-1]:.2f}%)")

    if labels is not None:
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.show()
