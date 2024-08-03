"""
Module definining base properties, attributes and methods of a model compatible with the
INTELLECT pipeline, meaning ranking/pruning/learning techniques.
"""
from abc import ABC, abstractmethod
from typing import Any

from river.base import DriftDetector

from ..io import dump, load


class BaseModel(ABC):
    """BaseModel class to define methods that should be required by
    subclasses in order to run the entire methodology.
    """
    @abstractmethod
    def __init__(self, drift_detector: DriftDetector = None) -> None:
        if drift_detector is not None and hasattr(drift_detector, 'clone'):
            drift_detector = drift_detector.clone()
        self.drift_detector = drift_detector

    @abstractmethod
    def learn(self, X: list, y: list, *args, **kwargs) -> tuple[list[float], float]:
        """Method to perform a single learning step

        Args:
            X (list): input data
            y (list): target labels

        Returns:
            tuple[list[float], float]: tuple with predictions and loss value
        """

    @abstractmethod
    def predict(self, X: list, *args, **kwargs) -> list[int]:
        """Function to perform prediction on provided data.

        Args:
            x (list): provided data to predict

        Returns:
            list[int]: list of prediction targets
        """

    @abstractmethod
    def predict_proba(self, X: list, *args, **kwargs) -> list[float]:
        """Function to perform predictions of probabilities

        Args:
            X (list): input data

        Returns:
            list[float]: list of probabilities
        """

    @abstractmethod
    def fit(self, *args, **kwargs) -> dict[str, list[float]] | None:
        """Function to fit the Model.

        Returns:
            dict[str, list[float]] | None: None or potentially the history dictionary
        """

    @abstractmethod
    def clone(self, init: bool = True) -> 'BaseModel':
        """Function to clone the model.

        Args:
            init (bool, optional): true whether to initialize a new model.
                Defaults to True.

        Returns:
            BaseModel: the new model
        """

    @abstractmethod
    def continuous_learning(self, *args, **kwargs) -> tuple[list[int], list[int], list[int]]:
        """Function to perform continuous learning on the provided data.

        Returns:
            tuple[list[int], list[int], list[int]]: tuple containing list of predictions, true values
                and the list of drifts, if any.
        """

    @abstractmethod
    def is_concept_drift(self, *args, **kwargs) -> bool:
        """Property to check if concept drift in the model has been detected.

        Returns:
            bool: whether a concept drift has been detected
        """

    @property
    @abstractmethod
    def prunable(self) -> list[str] | list[object]:
        """Property providing the list of the prunable layers in the model

        Returns:
            list[str] | list[object]: list of prunable layers.
        """

    def learn_one(self, x: list, y: list | float, *args, **kwargs) -> tuple[float, float]:
        """Function to perform one step in the learning process of a single sample

        Args:
            x (list): sample data
            y (list | float): sample label

        Returns:
            tuple[float, float]: predicted value and loss value
        """
        self.learn(x, y, *args, **kwargs)

    def predict_one(self, X: list, *args, **kwargs) -> int:
        """Function to predict provided sample

        Args:
            x (list): sample

        Returns:
            int: inferred label
        """
        return self.predict(X, *args, **kwargs)[0]

    def predict_proba_one(self, X: list, *args, **kwargs) -> float | list[float]:
        """Function to predict probabilities of a single sample

        Args:
            x (list): sample to predict

        Returns:
            float: probability/ies
        """
        return self.predict_proba(X, *args, **kwargs)[0]

    def save(self, path: str) -> Any:
        """Function to dump to file the current model.

        Args:
            path (str): the path of the file

        Returns:
            Any: the result of the dump procedure, usually None.
        """
        return dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Function to load from a given path the model.

        Args:
            path (str): path where the model is stored

        Returns:
            BaseModel: an instance of the model loaded
        """
        return load(path)
