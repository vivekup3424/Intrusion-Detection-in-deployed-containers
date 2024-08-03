"""
Module to provide functions for ranking the importance of features in a torch-based neural network.
"""
import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion

from ...dataset import Dataset
from .model import TorchModel


def rank_gradient_captum(model: TorchModel, data: Dataset,
                         current_features: list[str] = None, **kwargs) -> dict[str, float]:
    """Compute feature importance with integrated gradient method using captum

    Args:
        model (TorchModel): the target model
        data (Dataset): data for computing rank
        current_features (list[str], optional): list of available features to keep. Defaults to None.

    Returns:
        dict[str, float]: dictionary with name and rank for each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    if torch.cuda.is_available():
        model.model.cuda()
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    ig = IntegratedGradients(model)
    X = model.safe_cast_input(data.X)
    X.requires_grad_()
    attr = ig.attribute(X, target=1, **kwargs)
    attr = attr.detach().numpy()
    importances = np.mean(attr, axis=0)
    if torch.cuda.is_available():
        model.model.cpu()
    return dict(zip(current_features, [importances[i] for i in positions]))


def rank_perturbation_captum(model: TorchModel, data: Dataset,
                             current_features: list[str] = None, **kwargs) -> dict[str, float]:
    """Compute feature importance with occlusion method using captum

    Args:
        model (TorchModel): the target model
        data (Dataset): data for computing rank
        current_features (list[str], optional): list of available features to keep. Defaults to None.

    Returns:
        dict[str, float]: dictionary with name and rank for each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    if torch.cuda.is_available():
        model.model.cuda()
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(torch.tensor(data.X.values, device=model.device), **kwargs)
    attr = attributions_occ.detach().cpu().numpy()
    importances = np.mean(attr, axis=0)
    if torch.cuda.is_available():
        model.model.cpu()
    return dict(zip(current_features, [importances[i] for i in positions]))
