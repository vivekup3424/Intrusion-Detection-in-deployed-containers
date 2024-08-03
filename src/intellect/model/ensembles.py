"""
Module providing compatibility for river Ensembles jointly with local defined models.
"""
from river.base import Ensemble
from river.ensemble import LeveragingBaggingClassifier

from intellect.model.sklearn.model import BaseEnhancedMlp

from .base import BaseModel


class WrapRiverEnsemble(BaseModel):
    """Wrapper for the ensemble class when using a local model (e.g., torch.model.Mlp)
    with a river Ensemble. This class adds few methods to make it compatible and usable.
    Note that this is experimental, few functionalities might not work as they are not implemented.
    """

    def __init__(self, cls: Ensemble, *args, **kwargs):
        super().__init__()
        self.item: Ensemble = cls(*args, **kwargs)
        self.cnt = 0
        self.drifts = []

    @property
    def prev_drifts(self) -> int:
        """Property to return the number of previous concept drift recorded.

        Returns:
            int: number of previous drifts
        """
        if self.item.__class__ == LeveragingBaggingClassifier:
            return self.item.n_detected_changes
        raise ValueError()

    def learn(self, X: list, y: list, *args, **kwargs) -> tuple[list[float], float]:
        prev_drifts = self.prev_drifts
        for i in range(len(X)):
            self.item.learn_one(X.iloc[i:i+1], y.iloc[i:i+1].to_numpy(), *args, **kwargs)
            if self.is_concept_drift(prev_drifts):
                self.drifts.append(self.cnt + i)
        self.cnt += len(X)

    def predict(self, X: list, *args, **kwargs) -> list[int]:
        return [self.item.predict_one(X.iloc[i:i+1], as_dict=True) for i in range(len(X))]

    def predict_proba(self, X: list, *args, **kwargs) -> list[float]:
        return [self.item.predict_proba_one(X.iloc[i:i+1], as_dict=True) for i in range(len(X))]

    def fit(self, *args, **kwargs) -> dict[str, list[float]] | None:
        raise NotImplementedError()

    def clone(self, init: bool = True) -> 'BaseModel':
        raise NotImplementedError()

    def is_concept_drift(self, prev_concept, *args, **kwargs) -> bool:
        if self.item.__class__ == LeveragingBaggingClassifier:
            return self.item.n_detected_changes != prev_concept
        raise ValueError()

    @property
    def prunable(self) -> list[str] | list[object]:
        return []

    continuous_learning = BaseEnhancedMlp.continuous_learning
