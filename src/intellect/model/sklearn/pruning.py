"""
Module containing utility functions for pruning sklearn defined neural networks.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .model import BaseEnhancedMlp


def distance_l2(model: BaseEnhancedMlp, other: BaseEnhancedMlp, only_prunable: bool = True) -> float:
    """Return the norm 2 difference between the two model parameters

    Args:
        model (BaseEnhancedMlp): one model
        other (BaseEnhancedMlp): the other model
        only_prunable (bool, optional): whether to consider only prunable layers.
            Defaults to True.

    Returns:
        float: the l2 norm
    """
    t_own = model.prunable if only_prunable else model.parameters()
    t_other = other.prunable if only_prunable else other.parameters()
    ret = 0
    for i_lay_lvm, ii_lay_brm in zip(t_own, t_other):
        ret += np.sum((other.coefs_[ii_lay_brm] - model.coefs_[i_lay_lvm])**2).item()
        ret += np.sum((other.intercepts_[ii_lay_brm] - model.intercepts_[i_lay_lvm])**2).item()
    return ret


def sparsity(model: BaseEnhancedMlp, only_prunable: bool = True) -> tuple[float, list[int, float]]:
    """Function to compute the sparsity of a network

    Args:
        model (BaseEnhancedMlp): target network

    Returns:
        tuple[float, list[int, float]]: tuple with global and per-layer sparsity
    """
    if model.prune_masks is None:
        return 0, []
    layers = model.prunable if only_prunable else range(len(model.coefs_))
    single = [
        (np.sum(model.intercepts_[i] == 0) + np.sum(model.prune_masks[i] == 0) / model.prune_masks[i].size)
        for i in layers]
    return np.mean(single), single


def _globally_unstructured_connections(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable=True, cond='l1') -> BaseEnhancedMlp:
    check_is_fitted(model, msg='Model not fitted, fit before pruning')
    model = model.clone(init=False)
    model.prune_masks = [np.ones_like(k) for k in model.coefs_]
    pruned_idx = model.prunable if only_prunable else list(range(len(model.coefs_)))
    all_weights = np.concatenate([np.asarray(model.coefs_[i]).reshape(-1) for i in pruned_idx], axis=0)
    k = round(len(all_weights) * prune_ratio)
    if cond == 'random':
        idx = np.random.choice(all_weights.size, k, replace=False)
    else:
        if cond == 'l2':
            all_weights = np.sqrt(all_weights**2)
        elif cond == 'l1':
            all_weights = np.absolute(all_weights)
        else:
            raise NotImplementedError('Condition not implemented')
        idx = all_weights.argsort()[:k]
    mask = np.ones_like(all_weights)
    mask[idx] = 0
    pointer = 0
    for i in pruned_idx:
        num_param = model.coefs_[i].size
        model.prune_masks[i] = mask[pointer: pointer + num_param].reshape(model.coefs_[i].shape)
        pointer += num_param
    return model

def globally_unstructured_connections_l1(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable=True) -> BaseEnhancedMlp:
    return _globally_unstructured_connections(model, prune_ratio, only_prunable=only_prunable, cond='l1')

def globally_unstructured_connections_l2(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable=True) -> BaseEnhancedMlp:
    return _globally_unstructured_connections(model, prune_ratio, only_prunable=only_prunable, cond='l2')

def globally_unstructured_connections_random(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable=True) -> BaseEnhancedMlp:
    return _globally_unstructured_connections(model, prune_ratio, only_prunable=only_prunable, cond='random')


def _globally_neurons(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable: bool = True, cond='l2') -> BaseEnhancedMlp:
    """Function to prune CONNECTION-UNSTRUCTURED with L1 norm

    Args:
        model (BaseEnhancedMlp): model to be pruned
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        BaseEnhancedMlp: the pruned model (original one with in-place modification)
    """
    check_is_fitted(model, msg='Model not fitted, fit before pruning')
    model = model.clone(init=False)
    model.prune_masks = [np.ones_like(k) for k in model.coefs_]
    pruned_idx = model.prunable if only_prunable else list(range(len(model.coefs_)))

    if cond == 'random':
        all_weights = sum(model.coefs_[i].shape[1] for i in pruned_idx)
        k = round(all_weights * prune_ratio)
        idx = np.random.choice(all_weights, k, replace=False)
    else:
        all_weights = []
        for i in pruned_idx:
            for n_neuron in range(model.coefs_[i].shape[1]):
                if cond == 'l1':
                    all_weights.append(np.abs(model.coefs_[i][:, n_neuron]).sum())
                elif cond == 'l2':
                    all_weights.append(np.sqrt((model.coefs_[i][:, n_neuron] ** 2).sum()))
                else:
                    raise NotImplementedError('Condition not implemented')
        k = round(len(all_weights) * prune_ratio)
        idx = np.array(all_weights).argsort()[:k]

    prev = 0
    for i in pruned_idx:
        for n_neuron in range(model.coefs_[i].shape[1]):
            if prev + n_neuron in idx:
                model.prune_masks[i][:, n_neuron] = 0.
        prev += model.prune_masks[i].shape[1]
    return model

def globally_neurons_l2(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable: bool = True) -> BaseEnhancedMlp:
    return _globally_neurons(model, prune_ratio, only_prunable, cond='l2')

def globally_neurons_l1(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable: bool = True) -> BaseEnhancedMlp:
    return _globally_neurons(model, prune_ratio, only_prunable, cond='l1')

def globally_neurons_random(
        model: BaseEnhancedMlp, prune_ratio: float, only_prunable: bool = True) -> BaseEnhancedMlp:
    return _globally_neurons(model, prune_ratio, only_prunable, cond='random')
