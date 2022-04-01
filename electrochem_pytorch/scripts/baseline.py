import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from functools import partial

from .data_processing import get_class_label_to_int_mapping

CLASS_LABEL_TO_INT_MAPPING = get_class_label_to_int_mapping()


class RandomClassifier:
    """Returns random predictions of ints in the range 0-3."""

    def __init__(self, seed=0):
        np.random.seed(seed)
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        # Return array len(X) containing ints 0-3 randomly chosen
        return np.random.randint(0, 4, len(X))


class MaxPeakClassifier:
    """Returns predictions based off of the maximum peak value of the sample."""

    def fit(self, X, y=None):
        """
        Blank method to enable use with sklearn API.
        """
        return self

    def predict(self, X, y=None):
        """Predict classes 0-3 for X based off of the max peak value.

        From EDA, we know that the peaks for each analyte occur in the
        following index ranges:
        - Cd - (530, 580)
        - Cu - (660, 720)
        - Pb - (580, 620)

        This method checks the ranges and predicts the analyte with the
        highest value. However, if the average value of each peak is less
        than 2.5, it predicts seawater (this value was found through EDA
        too).

        Parameters
        ----------
        X : np.ndarray or pd.Series or pd.DataFrame
            The object containing voltage series for each analyte.
            Should be of shape (n_samples, 1002) since each series has len
            1002. Minimum length needed is 720 as that is the final peak
            index position that is checked.

            Note: you can also input a single series with shape (1002,).
        y : None, optional
            Placeholder for y to maintain compatibility with sklearn, by default None

        Returns
        -------
        np.ndarray
            1D array containing predictions for each sample as ints.
            The mapping is:
                Cd - 0
                Cu - 1
                Pb - 2
                Sw - 3
        """
        # Turn Series and DataFrames into NumPy arrays
        if not isinstance(X, np.ndarray):
            X = X.values
        if X.ndim == 1:
            # Reshape single sample 1D inputs to 2D
            X = X.reshape(1, -1)
        # Define peak value ranges
        cu_peak_start = 660
        cu_peak_end = 720

        cd_peak_start = 530
        cd_peak_end = 580

        pb_peak_start = 580
        pb_peak_end = 620

        preds = []
        # Iterate over the rows (i.e. samples) in X
        for sample in X:
            # Get peak values from sample
            cu_peak_value = sample[cu_peak_start:cu_peak_end].max()
            cd_peak_value = sample[cd_peak_start:cd_peak_end].max()
            pb_peak_value = sample[pb_peak_start:pb_peak_end].max()

            peak_values = {
                "Cd": cd_peak_value,
                "Cu": cu_peak_value,
                "Pb": pb_peak_value,
            }

            avergae_value = np.mean(list(peak_values.values()))

            # On average, seawater max values are less than 2.5, found by inspection
            if avergae_value <= 2.5:
                pred = CLASS_LABEL_TO_INT_MAPPING["Sw"]
                preds.append(pred)
            else:
                analyte_with_biggest_peak = max(peak_values, key=peak_values.get)
                pred = CLASS_LABEL_TO_INT_MAPPING[analyte_with_biggest_peak]
                preds.append(pred)
        return np.array(preds)


def print_scores(model, X, y, average="micro"):
    """Print accuracy, precision, recall and f1 scores for a given
    fit model.

    Parameters
    ----------
    model : A fitted model with a predict() method
        A model that has already been trained on data and is ready
        to make predictions
    X : np.ndarray, pd.Series, pd.DataFrame
        Input array containing samples with shape (n_samples, n_features)
    y : np.ndarray or pd.Series
        True predictions
    average : str, optional
        Determines the averaging performed on the data. Required for
        multiclass classification. Options are: 'micro' (default),
        ‘micro’, ‘macro’, ‘samples’, ‘weighted’.
    """
    preds = model.predict(X)

    scores = {
        "accuracy": accuracy_score,
        "f1": partial(f1_score, average="micro"),
        "precision": partial(precision_score, average="micro"),
        "recall": partial(recall_score, average="micro"),
    }

    for name, score_func in scores.items():
        if name != "accuracy":
            print(f"{name}_{average} - {score_func(y, preds):.4f}")
        else:
            print(f"{name} - {score_func(y, preds):.4f}")


def print_kfold_scores(model, X, y, n_splits=5, **kwargs):
    """Runs KFold validation n_splits times and prints the accuracy,
    f1, precision and recall scores for each fold.

    Parameters
    ----------
    model : Model with fit and predict methods
        The model to use for training and predicting
    X : np.ndarray, pd.Series, pd.DataFrame
        Input array containing samples with shape (n_samples, 1002)
    y : np.ndarray or pd.Series
        Output array
    n_splits : int
        The number of folds to create, defaults to 5
    """
    kfold = KFold(n_splits=n_splits)
    fold = 1
    for train_idx, test_idx in kfold.split(X):
        print(f"FOLD {fold}")
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]

        model.fit(X_train, y_train)

        print_scores(model, X_test, y_test, **kwargs)
        print()
        fold += 1
