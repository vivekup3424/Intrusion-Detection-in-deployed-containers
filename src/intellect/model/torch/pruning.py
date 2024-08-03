"""
Module containing utility functions for pruning torch-based previously defined neural networks.
"""
from collections.abc import Iterable
from enum import Enum

import pandas as pd
import torch
from torch.nn.utils import prune

from .interpretability import get_neurons_activation
from .model import TorchModel


class PruningConstants(Enum):
    """Pruning constants helper."""

    PRUNE_CONNECTIONS_DIM = 1
    PRUNE_NEURONS_DIM = 0

    RANDOM = 0
    L1_NORM = 1
    L2_NORM = 2


###################################
# LOCAL PRUNING
###################################


def _helper_locally(model: TorchModel, prune_ratio: float, n: int = PruningConstants.L1_NORM.value,
                    dim: int = PruningConstants.PRUNE_NEURONS_DIM.value,
                    structured: bool = False, name: str = 'weight', **kwargs) -> TorchModel:
    """Helper internal function to perform the desired pruning algorithm.

    Args:
        model (TorchModel): original model to be pruned.
        prune_ratio (_type_): pruning ratio (the higher the more is removed).
        n (int, optional): whether to prune random (0), l1 (1) or l2 (2) norm.
            Defaults to PruningConstants.L1_NORM.value.
        dim (int, optional): whether to prune neurons (0) or connections (1).
            Defaults to PruningConstants.PRUNE_NEURONS_DIM.value.
        structured (bool, optional): whether to prune structured (entire neuron/connections) or sparse.
            Defaults to False.
        name (str, optional): pruning parameter name within 'weight' or 'bias'. Defaults to "weight".

    Raises:
        NotImplementedError: if structured and requiring a different norm than l1 or l2
        NotImplementedError: if unstructured and requiring a different norm than l1 or l2

    Returns:
        TorchModel: the pruned version of the model
    """
    if len(kwargs):
        print('Unused', kwargs)
    model = model.clone(init=False)
    layers = model.prunable
    # dim=0 prune neuron, dim=1 prune connection.
    # when called with dim=0 it produces the same output of unstructured
    if structured:
        if n == PruningConstants.RANDOM.value:
            for k, v in zip(layers, prune_ratio):
                prune.random_structured(k, name, v, dim)
        elif n in (PruningConstants.L1_NORM.value, PruningConstants.L2_NORM.value):
            for k, v in zip(layers, prune_ratio):
                prune.ln_structured(k, name, amount=v, n=n, dim=dim)
        else:
            raise NotImplementedError(f'Norm n {n} not implemented')
    else:
        if n == PruningConstants.RANDOM.value:
            for k, v in zip(layers, prune_ratio):
                prune.random_unstructured(k, name, v)
        elif n == PruningConstants.L1_NORM.value:
            for k, v in zip(layers, prune_ratio):
                prune.l1_unstructured(k, name, v)
        else:
            raise NotImplementedError(f'Norm n {n} not implemented')
    return model


def _helper_locally_structured_neurons_activation(
        model: TorchModel, prune_ratio: float, X: pd.DataFrame, n=PruningConstants.L1_NORM.value,
        name='weight', **kwargs) -> TorchModel:
    """Function internal to handle locally structured neuron activations pruning methods

    Args:
        model (TorchModel): target model
        prune_ratio (float): pruning ratio between 0 and 1
        X (pd.DataFrame): input data
        n (_type_, optional): l1, l2 or random method. Defaults to PruningConstants.L1_NORM.value.
        name (str, optional): name of the parameters to prune. Defaults to "weight".

    Returns:
        TorchModel: _description_
    """
    if len(kwargs):
        print('Unused', kwargs)
    model = model.clone(init=False)

    activations = get_neurons_activation(model, X, only_prunable=True)

    for lay, a in activations.items():
        prune.ln_structured(lay, name=name, amount=prune_ratio, n=n,
                            dim=PruningConstants.PRUNE_NEURONS_DIM.value, importance_scores=a)
    return model


def locally_structured_neurons_l1(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune NEURONS-LOCALLY-STRUCTURED using l1 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.L1_NORM.value,
                           dim=PruningConstants.PRUNE_NEURONS_DIM.value, structured=True, **kwargs)


def locally_structured_neurons_l2(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune NEURONS-LOCALLY-STRUCTURED using l2 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.L2_NORM.value,
                           dim=PruningConstants.PRUNE_NEURONS_DIM.value, structured=True ** kwargs)


def locally_structured_neurons_random(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune NEURONS-LOCALLY-STRUCTURED random

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.RANDOM.value,
                           dim=PruningConstants.PRUNE_NEURONS_DIM.value, structured=True, **kwargs)


def locally_structured_connections_l1(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-LOCALLY-STRUCTURED using l1 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.L1_NORM.value,
                           dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, structured=True, **kwargs)


def locally_structured_connections_l2(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-LOCALLY-STRUCTURED using l2 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.L2_NORM.value,
                           dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, structured=True ** kwargs)


def locally_structured_connections_random(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-LOCALLY-STRUCTURED random

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.RANDOM.value,
                           dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, structured=True, **kwargs)


def locally_unstructured_connections_l1(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-LOCALLY-UNSTRUCTURED using l1 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.L1_NORM.value,
                           dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, structured=False, **kwargs)


def locally_unstructured_connections_random(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-LOCALLY-UNSTRUCTURED random

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally(model, prune_ratio, n=PruningConstants.RANDOM.value,
                           dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, structured=False, **kwargs)


def locally_structured_neurons_activation_l1(
        model: TorchModel, prune_ratio: float, X: pd.DataFrame, **kwargs) -> TorchModel:
    """Function to prune NEURONS-LOCALLY-STRUCTURED using l1

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally_structured_neurons_activation(
        model, prune_ratio, X, n=PruningConstants.L1_NORM.value, **kwargs)


def locally_structured_neurons_activation_l2(
        model: TorchModel, prune_ratio: float, X: pd.DataFrame, **kwargs) -> TorchModel:
    """Function to prune NEURONS-LOCALLY-STRUCTURED using l2

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_locally_structured_neurons_activation(
        model, prune_ratio, X, n=PruningConstants.L2_NORM.value, **kwargs)


###################################
# GLOBAL PRUNING
###################################

def _torch_missing_globally_structured(
        parameters, pruning_method, prune_ratio,
        dim=PruningConstants.PRUNE_NEURONS_DIM.value, importance_scores=None, **kwargs) -> TorchModel:
    """Function internal to prune globally structured (still missing in PyTorch).

    Args:
        parameters (_type_): iterable of parameters to prune
        pruning_method (_type_): pruning method to be used
        prune_ratio (_type_): pruning ratio between 0 and 1
        dim (_type_, optional): whether to prune connections or neurons.
            Defaults to PruningConstants.PRUNE_NEURONS_DIM.value.
        importance_scores (_type_, optional): importance score for parameters. Defaults to None.

    Raises:
        TypeError: if parameters is not iterable
        TypeError: if importance_scores is not a dictionary
        TypeError: if pruning method is not structured

    Returns:
        TorchModel: _description_
    """
    if not isinstance(parameters, Iterable):
        raise TypeError('global_unstructured(): parameters is not an Iterable')

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError('global_unstructured(): importance_scores must be of type dict')

    to_look = int(not dim)

    def find_max_shape():
        max_dim = 0
        for (module, name) in parameters:
            v = importance_scores.get((module, name), getattr(module, name))
            max_dim = max(max_dim, v.shape[to_look])
        return max_dim

    max_dim = find_max_shape()

    def tensor_to_shape(t: torch.Tensor):
        new_shape = list(t.shape)
        new_shape[to_look] = max_dim
        new_shape = tuple(new_shape)
        tmp = torch.zeros(new_shape)
        tmp[:t.shape[0], :t.shape[1]] = t
        return tmp

    relevant_importance_scores = torch.cat(
        [
            tensor_to_shape(importance_scores.get((module, name), getattr(module, name)))
            for (module, name) in parameters
        ], dim
    )

    default_mask = torch.cat(
        [
            tensor_to_shape(getattr(module, name + '_mask', torch.ones_like(getattr(module, name))))
            for (module, name) in parameters
        ], dim
    )

    container = prune.PruningContainer()
    container._tensor_name = 'temp'
    method = pruning_method(prune_ratio, dim=dim, **kwargs)
    method._tensor_name = 'temp'
    if method.PRUNING_TYPE != 'structured':
        raise TypeError(
            'Only "structured" PRUNING_TYPE supported for '
            f'the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}')

    container.add_pruning_method(method)

    final_mask = container.compute_mask(relevant_importance_scores, default_mask)
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        if dim == 0:
            param_mask = final_mask[pointer: pointer + param.shape[0], :param.shape[1]]
        if dim == 1:
            param_mask = final_mask[:param.shape[0], pointer: pointer + param.shape[1]]
        pointer += param.shape[dim]
        prune.custom_from_mask(module, name, mask=param_mask)


def _helper_globally(model: TorchModel, prune_ratio: float, method=prune.L1Unstructured,
                     dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, name='weight', **kwargs) -> TorchModel:
    """Helper function to handle globally pruning methods

    Args:
        model (TorchModel): target model
        prune_ratio (float): pruning ratio between 0 and 1
        method (_type_, optional): pruning method to use. Defaults to prune.L1Unstructured.
        dim (_type_, optional): prune connections or neurons. Defaults to PruningConstants.PRUNE_CONNECTIONS_DIM.value.
        name (str, optional): name of the parameter to prune. Defaults to "weight".

    Returns:
        TorchModel: _description_
    """
    model = model.clone(init=False)
    layers = model.prunable
    parameters = {(x, name) for x in layers}

    if method.PRUNING_TYPE == 'structured':
        _torch_missing_globally_structured(parameters, method, prune_ratio, dim=dim, **kwargs)
    else:
        prune.global_unstructured(parameters,
                                  pruning_method=method,
                                  amount=prune_ratio)
    return model


def _helper_globally_structured_neurons_activation(model: TorchModel, prune_ratio: float, X: pd.DataFrame,
                                                   n, name='weight', **kwargs) -> TorchModel:
    """Helper internal function for handling global structured neurons activation pruning methods.

    Args:
        model (TorchModel): target model
        prune_ratio (float): pruning ratio between 0 and 1
        X (pd.DataFrame): input data
        n (_type_): norm type
        name (str, optional): name of the parameter to prune. Defaults to "weight".

    Returns:
        TorchModel: the pruned model
    """
    layers = model.prunable

    activations = model.get_neurons_activation(layers, X, n, shape_fill=name)

    return _helper_globally(model, prune_ratio, method=prune.LnStructured,
                            dim=PruningConstants.PRUNE_NEURONS_DIM.value, n=n,
                            importance_scores=activations, **kwargs)


def globally_structured_neurons_l1(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune NEURONS-GLOBALLY-STRUCTURED using l1 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.LnStructured, n=PruningConstants.L1_NORM.value,
        dim=PruningConstants.PRUNE_NEURONS_DIM.value, **kwargs)


def globally_structured_neurons_l2(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune NEURONS-GLOBALLY-STRUCTURED using l2 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.LnStructured, n=PruningConstants.L2_NORM.value,
        dim=PruningConstants.PRUNE_NEURONS_DIM.value, **kwargs)


def globally_structured_neurons_random(
        model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune NEURONS-GLOBALLY-STRUCTURED random

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.RandomStructured, dim=PruningConstants.PRUNE_NEURONS_DIM.value, **kwargs)


def globally_structured_connections_l1(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-GLOBALLY-STRUCTURED using l1 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.LnStructured, n=PruningConstants.L1_NORM.value,
        dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, **kwargs)


def globally_structured_connections_l2(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-GLOBALLY-STRUCTURED using l2 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.LnStructured, n=PruningConstants.L2_NORM.value,
        dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, **kwargs)


def globally_structured_connections_random(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-GLOBALLY-STRUCTURED random

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.LnStructured, n=PruningConstants.RANDOM.value,
        dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, **kwargs)


def globally_unstructured_connections_l1(model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-GLOBALLY-UNSTRUCTURED using l1 norm

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.L1Unstructured, n=PruningConstants.L1_NORM.value,
        dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, **kwargs)


def globally_unstructured_connections_random(
        model: TorchModel, prune_ratio: float, **kwargs) -> TorchModel:
    """Function to prune CONNECTIONS-GLOBALLY-UNSTRUCTURED random

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally(
        model, prune_ratio, method=prune.RandomUnstructured, n=PruningConstants.RANDOM.value,
        dim=PruningConstants.PRUNE_CONNECTIONS_DIM.value, **kwargs)


def globally_structured_neurons_activation_l1(
        model: TorchModel, prune_ratio: float, X: pd.DataFrame, **kwargs) -> TorchModel:
    """Function to prune NEURON-GLOBALLY-STRUCTURED by l1 norm of neuron activations

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1
        X (pd.DataFrame): input data

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally_structured_neurons_activation(
        model, prune_ratio, X, n=PruningConstants.L1_NORM.value, **kwargs)


def globally_structured_neurons_activation_l2(
        model: TorchModel, prune_ratio: float, X: pd.DataFrame, **kwargs) -> TorchModel:
    """Function to prune NEURON-GLOBALLY-STRUCTURED by l2 norm of neuron activations

    Args:
        model (TorchModel): target model
        prune_ratio (float): prune ratio between 0 and 1
        X (pd.DataFrame): input data

    Returns:
        TorchModel: the pruned model
    """
    return _helper_globally_structured_neurons_activation(
        model, prune_ratio, X, n=PruningConstants.L2_NORM.value, **kwargs)
