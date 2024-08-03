"""
Module to provide utility functions for debugging/inspecting previously defined neural networks.
"""
import time

import numpy as np
import torch
from captum.attr import LayerActivation, LayerConductance, NeuronConductance
from torch.nn.utils import prune
from tqdm import tqdm

from ...dataset import Dataset
from .model import TorchModel

# the feature ranking methods are still usable for interpretability


def get_neurons_activation(model: TorchModel, data: Dataset,
                           only_prunable: bool = True, **kwargs) -> dict[torch.nn.Module, torch.tensor]:
    """Function to compute neurons' activation of all the layers.
    Example at: https://captum.ai/tutorials/Titanic_Basic_Interpret

    Args:
        model (TorchModel): target model
        data (Dataset): input data
        only_prunable (bool, optional): whether to consider only prunable. Defaults to True.

    Returns:
        dict[torch.nn.Module, torch.tensor]: a tensor of activations for each layer
    """

    model.cuda()
    layers = model.prunable if only_prunable else tuple(model.layers.children())
    ig = LayerActivation(model, layers)
    X = model.safe_cast_input(data.X)
    X.requires_grad_()
    attr = ig.attribute(X, **kwargs)
    model.cpu()
    return dict(zip(layers, attr))


def get_neurons_importance(
        model: TorchModel, data: Dataset,
        only_prunable: bool = True, **kwargs) -> dict[torch.nn.Module, list[float]]:
    """Function to return neurons importance for each layer

    Args:
        model (TorchModel): target model
        data (Dataset): input data
        only_prunable (bool, optional): true whether to consider only prunable. Defaults to True.

    Returns:
        dict[torch.nn.Module, list[float]]: dict with list of neurons' importances for each layer
    """
    model.cuda()
    layers = model.prunable if only_prunable else tuple(model.layers.children())
    X = model.safe_cast_input(data.X)
    ret = {}
    for layer in layers:
        cond = LayerConductance(model, layer)
        ret[layer] = np.mean(cond.attribute(X, **kwargs).detach().cpu().numpy(), axis=0)
    model.cpu()
    return ret


def get_learned_features_per_neuron(
        model: TorchModel, data: Dataset,
        target_layer: torch.nn.Module, **kwargs) -> list[list[float]]:
    """Function to return a score assigned to the learned features for each neuron in a specific layer of a network

    Args:
        model (TorchModel): the model to be used
        data (Dataset): the input data
        target_layer (torch.nn.Module): target layer to compute

    Returns:
        list[torch.nn.Module, list[float]]: list with list of neurons' list of feature importances
    """
    model.cuda()
    X = model.safe_cast_input(data.X)
    ret = []
    for i in range(next(target_layer.parameters()).shape[0]):
        neuron_cond = NeuronConductance(model, target_layer)
        ret.append(np.mean(neuron_cond.attribute(X, neuron_selector=i, **kwargs).detach().cpu().numpy(), axis=0))
    model.cpu()
    return ret


def benchmark_forward(model: TorchModel, X: list, times: int, warmup: int) -> list[int]:
    """Function to benchmark the time execution of the forward function.

    Args:
        model (TorchModel): the model of interest
        X (list): the input data
        times (int): number of iterations
        warmup (int): number of warmup iterations

    Returns:
        list[int]: list of times
    """
    scores = []
    with torch.no_grad():
        X = model.safe_cast_input(X)
        for _ in tqdm(range(warmup)):
            model.model(X)
        for _ in tqdm(range(times)):
            start = time.time_ns()
            model.model(X)
            end = time.time_ns()
            scores.append(len(X) / (end - start))
    return scores


def network_layers_sparsity(model: TorchModel,
                            only_prunable: bool = True) -> tuple[float, dict[torch.nn.Module, float]]:
    """Function to compute the model sparsity, hence the ratio of parameters equal to 0.

    Args:
        model (TorchModel): the model of interest
        only_prunable (bool, optional): true whether to consider only prunable layers. Defaults to True.

    Returns:
        tuple[float, list[torch.nn.Module, float]]: tuple with global sparsity and per-layer sparsity
    """
    if prune.is_pruned(model.model):
        model = model.clone(init=False)
    single = {}
    total = 0
    target = model.prunable if only_prunable else model.model.children()
    for x in target:
        local = 0
        for name, p in x.named_parameters():
            data = p.data
            if name.endswith('_orig'):
                name = name.replace('_orig', '_mask')
                data = data * getattr(x, name).data
            local += float(torch.sum(data == 0.) / p.data.nelement())
        single[x] = local
        total += local
    total /= len(single)
    return total, single


def distance_l2(model: TorchModel, other: TorchModel, only_prunable=True) -> float:
    """Return the norm 2 difference between the two model parameters

    Args:
        model (TorchModel): one model
        other (TorchModel): the other model
        only_prunable (bool, optional): true if computed only on the prunable layers.
            Defaults to True.

    Returns:
        float: the l2 distance
    """
    t_own = model.prunable if only_prunable else model.parameters()
    t_other = other.prunable if only_prunable else other.parameters()
    ret = 0
    for lay_lvm, lay_brm in zip(t_own, t_other):
        for k, v in lay_brm.named_parameters():
            vv = getattr(lay_lvm, k)
            ret += torch.sum((v - vv)**2).item()
    return ret
