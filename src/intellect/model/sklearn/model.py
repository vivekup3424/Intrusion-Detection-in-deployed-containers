"""
Module definining the enhanced version of sklearn multilayer perceptron neural
networks with additional support to the dropout and pruning methodologies.
"""
import math
import sys
from copy import deepcopy
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
import sklearn
from numpy import ndarray
from numpy.random import RandomState
from river.base import DriftDetector
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import (ACTIVATIONS, DERIVATIVES,
                                          LOSS_FUNCTIONS)
from sklearn.neural_network._multilayer_perceptron import \
    BaseMultilayerPerceptron
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from ...dataset import ContinuouLearningAlgorithm, Dataset, InputForLearn
from ...scoring import compute_metric_percategory
from ..base import BaseModel


class BaseEnhancedMlp(BaseMultilayerPerceptron, BaseModel):
    """Enhanced version of sklearn multilayer perceptron, with additional
    support to dropout and pruning techniques. This is an experimental
    class, some of the base functionalities are not implemented yet.
    """

    _parameter_constraints: dict = {
        **BaseMultilayerPerceptron._parameter_constraints,
        'drift_detector': [DriftDetector, None],
        'dropout': [Interval(Real, 0, 1, closed='left')],
        'prune_masks': [np.ndarray, list, None]
    }

    def __init__(
            self, classes: list[int],
            drift_detector=None, dropout: float = 0., prune_masks: list[np.ndarray] = None, hidden_layer_sizes=...,
            activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu', *,
            solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam', alpha: float = 0.0001, batch_size: int | str = 'auto',
            learning_rate_init: float = 0.001, power_t: float = 0.5, max_iter: int = 200,
            loss: str = None, shuffle: bool = True,
            learning_rate: Literal['constant', 'invscaling', 'adaptive'] = 'constant',
            random_state: int | RandomState | None = None, tol: float = 0.0001, verbose: bool = False,
            warm_start: bool = False, momentum: float = 0.9,
            nesterovs_momentum: bool = True, early_stopping: bool = False, validation_fraction: float = 0.1,
            beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8, n_iter_no_change: int = 10,
            max_fun: int = 15000):
        super().__init__(
            hidden_layer_sizes, activation, solver=solver, alpha=alpha, batch_size=batch_size,
            learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
            loss=loss, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,
            momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
            validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        BaseModel.__init__(self, drift_detector=drift_detector)
        self.classes = classes
        self.drift_detector = drift_detector
        self.dropout = dropout
        self.prune_masks = prune_masks

        # fix to make also a Regressor look like a classifier and at the same enable
        # knwoledge distillation
        self._estimator_type = 'classifier'
        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(self.classes)
        self.classes_ = self._label_binarizer.classes_

    def _compute_dropout_masks(self, X):
        dropout_masks = None
        layer_units = [X.shape[1]] + list(self.hidden_layer_sizes) + [self.n_outputs_]

        # Create the Dropout Mask (DROPOUT ADDITION)
        if self.dropout not in (0, None):
            if not 0 < self.dropout < 1:
                raise ValueError('Dropout must be between zero and one. If Dropout=X then, 0 < X < 1.')
            keep_probability = 1 - self.dropout
            dropout_masks = [np.ones(layer_units[0])]

            # Create hidden Layer Dropout Masks
            for units in layer_units[1:-1]:
                # Create inverted Dropout Mask, check for random_state
                if self.random_state is not None:
                    layer_mask = self._random_state.random(units) < keep_probability
                    layer_mask = layer_mask.astype(int) / keep_probability
                else:
                    layer_mask = (np.random.rand(units) < keep_probability).astype(int) / keep_probability
                dropout_masks.append(layer_mask)
        return dropout_masks

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads,):
        n_samples = X.shape[0]

        dropout_masks = self._compute_dropout_masks(X)

        # Forward propagate
        activations = self._forward_pass(activations, dropout_masks=dropout_masks)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        # Apply Dropout Masks to the Parameter Gradients (DROPOUT ADDITION)
        if dropout_masks is not None:
            for layer in range(len(coef_grads) - 1):
                mask = (~(dropout_masks[layer + 1] == 0)).astype(int)
                coef_grads[layer] = coef_grads[layer] * mask[None, :]
                coef_grads[layer + 1] = coef_grads[layer + 1] * mask.reshape(-1, 1)
                intercept_grads[layer] = intercept_grads[layer] * mask
        return loss, coef_grads, intercept_grads

    def _forward_pass(self, activations, dropout_masks=None):
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            tmp_coef = self.coefs_[i]
            if self.prune_masks:
                tmp_coef *= self.prune_masks[i]
            activations[i + 1] = safe_sparse_dot(activations[i], tmp_coef)
            activations[i + 1] += self.intercepts_[i]
            if (i + 1) != (self.n_layers_ - 1):
                if dropout_masks is not None:
                    activations[i + 1] = activations[i + 1] * dropout_masks[i + 1][None, :]
                hidden_activation(activations[i + 1])
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])
        return activations

    def _forward_pass_fast(self, X, check_input=True):
        if check_input:
            X = self._validate_data(X, accept_sparse=['csr', 'csc'], reset=False)
        activation = X
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            tmp_coef = self.coefs_[i]
            if self.prune_masks:
                tmp_coef *= self.prune_masks[i]
            activation = safe_sparse_dot(activation, tmp_coef)
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)
        return activation

    def clone(self, init=True, copy_prune=True):
        if not init:
            new = deepcopy(self)
        else:
            new = sklearn.base.clone(self, safe=True)
        if copy_prune:
            new.copy_prune(self)
        return new

    def copy_prune(self, other: BaseModel):
        """Function to copy the pruning mask of the provided model into the current one.

        Args:
            other (BaseModel): model to copy the prune mask from.
        """
        self.prune_masks = other.prune_masks

    def learn(self, X, y, *args, **kwargs):
        self.partial_fit(X, y, *args, **kwargs)

    def learn_knowledge_distillation(self, inputs, oracle_proba, true_labels,
                                     *args, alpha: float = 1., **kwargs):
        final_labels = alpha * oracle_proba + (1-alpha) * true_labels
        self.partial_fit(inputs, final_labels, *args, **kwargs)

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset | float = 0.2, batch_size: int = 256,
            max_epochs: int = 100, epochs_wo_improve: int = None, metric=accuracy_score, monitor_ds: Dataset = None,
            shuffle=True, higher_better=True, learn_input=InputForLearn.client, oracle=None, learn_kwargs=None,
            algorithm: ContinuouLearningAlgorithm = ContinuouLearningAlgorithm.ground_truth, verbose=False,
            idx_active_features=None, idx_active_features_oracle=None, concept_react_func=None):

        self.warm_start = False
        self.best_loss_ = sys.maxsize
        self.batch_size = batch_size
        self.n_iter_ = 0
        self.max_iter = max_epochs
        self.n_iter_no_change = max_epochs
        self.verbose = False
        self.shuffle = shuffle
        self.early_stopping = False
        self.validation_fraction = 0.

        if learn_kwargs is None:
            learn_kwargs = {}
        if idx_active_features is None:
            idx_active_features = []
        if idx_active_features_oracle is None:
            idx_active_features_oracle = []
        if metric is None:
            higher_better = False

        if isinstance(self, EnhancedMlpClassifier):
            learn_kwargs['classes'] = self.classes

        monitored_metrics = []
        best_metric = 0 if higher_better else sys.maxsize
        current_epochs_wo_improve = 0
        history = {'loss_train': []}

        best_state_dict = None

        if validation_dataset is not None and validation_dataset != 0:
            if isinstance(validation_dataset, float):
                validation_dataset = train_dataset.sample(validation_dataset, by_category=True)
                train_dataset = train_dataset.filter_indexes(validation_dataset.X.index.values)
            else:
                validation_dataset = validation_dataset.clone()
            history['loss_validation'] = []

        if metric is not None:
            history[f'{metric.__name__}_train'] = []
            if validation_dataset:
                history[f'{metric.__name__}_validation'] = []

        inputs = train_dataset.X
        labels = train_dataset.y
        inputs_client = inputs_oracle = inputs
        true_labels = labels

        if idx_active_features:
            inputs_client = inputs.copy()
            inputs_client.iloc[:, idx_active_features] = 0.
            if validation_dataset:
                validation_dataset.X.iloc[:, idx_active_features] = 0.
            if monitor_ds:
                monitor_ds = monitor_ds.clone()
                monitor_ds.X.iloc[:, idx_active_features] = 0.

        if idx_active_features_oracle:
            inputs_oracle = inputs.copy()
            inputs_oracle.iloc[:, idx_active_features_oracle] = 0.

        if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
            inferred_labels = oracle.predict(inputs_oracle)

        if algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
            teacher_logits = oracle.predict_proba(inputs_oracle)

        if learn_input.value == InputForLearn.oracle.value:
            inputs = inputs_oracle
        elif learn_input.value == InputForLearn.client.value:
            inputs = inputs_client
        elif learn_input.value == InputForLearn.mixed.value:
            inputs = pd.concat([inputs]*2, ignore_index=True)
            true_labels = pd.concat([true_labels] * 2, ignore_index=True)
            if algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                teacher_logits = teacher_logits.repeat(2)
            if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                inferred_labels = inferred_labels.repeat(2)
        else:
            raise ValueError(f'Unknown learn input {learn_input} {learn_input.value}')

        for _ in range(max_epochs) if not verbose else (pbar := tqdm(range(max_epochs))):

            if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                self.learn(inputs, inferred_labels, **learn_kwargs)
            elif algorithm.value == ContinuouLearningAlgorithm.ground_truth.value:
                self.learn(inputs, true_labels, **learn_kwargs)
            elif algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                self.learn_knowledge_distillation(
                    inputs, teacher_logits, true_labels, **learn_kwargs)
            else:
                raise ValueError(f'Unknown algorithm {algorithm} {algorithm.value}')

            if self.drift_detector is not None:
                for vt, vp in zip(labels, self.predict(inputs_client)):
                    self.drift_detector.update(int(not vt == vp))
                    if self.is_concept_drift():
                        if concept_react_func is not None:
                            concept_react_func(self)

            eval_metric = self.loss_curve_[-1]
            history['loss_train'].append(eval_metric)
            if metric:
                eval_metric = metric(labels, self.predict(inputs_client))
                history[f'{metric.__name__}_train'].append(eval_metric)

            if validation_dataset:
                eval_metric = LOSS_FUNCTIONS[self.loss](
                    validation_dataset.y, self.predict_proba(validation_dataset.X))
                history['loss_validation'].append(eval_metric)
                if metric:
                    eval_metric = metric(validation_dataset.y, self.predict(validation_dataset.X))
                    history[f'{metric.__name__}_validation'].append(eval_metric)

            if monitor_ds:
                monitored_metrics.append(compute_metric_percategory(
                    monitor_ds.y, self.predict(monitor_ds.X),
                    monitor_ds._y, scorer=metric or accuracy_score))

            if not epochs_wo_improve:
                continue

            if verbose:
                pbar.set_description(str({k: v[-1] for k, v in history.items()}))

            cond = eval_metric > best_metric if higher_better else eval_metric < best_metric
            current_epochs_wo_improve += 1
            if cond:
                current_epochs_wo_improve = 0
                best_metric = eval_metric
                best_state_dict = {'coefs_': deepcopy(self.coefs_), 'intercepts_': deepcopy(self.intercepts_)}

            if current_epochs_wo_improve == epochs_wo_improve:
                break

        if best_state_dict:
            for k, v in best_state_dict.items():
                setattr(self, k, v)

        return history, monitored_metrics

    def continuous_learning(
            self, ds: Dataset, epochs=1, batch_size=1, learn_input=InputForLearn.client, oracle=None,
            learn_kwargs=None, algorithm: ContinuouLearningAlgorithm = ContinuouLearningAlgorithm.ground_truth,
            verbose=False, idx_active_features=None, idx_active_features_oracle=None, shuffle=False,
            concept_react_func=None, **kwargs):

        if learn_kwargs is None:
            learn_kwargs = {}
        if idx_active_features is None:
            idx_active_features = []
        if idx_active_features_oracle is None:
            idx_active_features_oracle = []

        if isinstance(self, EnhancedMlpClassifier):
            learn_kwargs['classes'] = self.classes

        inputs = ds.X
        true_labels = ds.y
        inputs_client = inputs_oracle = inputs

        if idx_active_features:
            inputs_client = inputs.copy()
            inputs_client.iloc[:, idx_active_features] = 0.

        if idx_active_features_oracle:
            inputs_oracle = inputs.copy()
            inputs_oracle.iloc[:, idx_active_features_oracle] = 0.

        if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
            inferred_labels = oracle.predict(inputs_oracle)

        if algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
            teacher_logits = oracle.predict_proba(inputs_oracle)

        if learn_input.value == InputForLearn.oracle.value:
            inputs = inputs_oracle
        elif learn_input.value == InputForLearn.client.value:
            inputs = inputs_client
        elif learn_input.value == InputForLearn.mixed.value:
            batch_size *= 2
            inputs = np.vstack((inputs_client, inputs_oracle)).ravel('F')
            true_labels = pd.concat([true_labels] * 2, ignore_index=True)
            if algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                teacher_logits = teacher_logits.repeat(2)
            if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                inferred_labels = inferred_labels.repeat(2)

        self.batch_size = batch_size
        self.warm_start = False
        self.best_loss_ = sys.maxsize
        self.n_iter_ = 0
        self.max_iter = 1
        self.n_iter_no_change = 1
        self.verbose = False
        self.shuffle = shuffle
        self.early_stopping = False
        self.validation_fraction = 0.

        y_preds, drifts = np.empty(0), np.empty(0)
        niter = math.ceil(len(ds)/batch_size)

        for i, _ in enumerate(range(niter) if not verbose else tqdm(range(niter))):
            base = i*batch_size
            tmp_inputs = inputs.iloc[base: base+batch_size]
            tmp_true_labels = true_labels.iloc[base: base+batch_size]
            tmp_inputs_client = inputs_client.iloc[base: base+batch_size]

            pred = self.predict(tmp_inputs_client)

            y_preds = np.concatenate((y_preds, pred), axis=0)

            for _ in range(epochs):
                if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                    self.learn(tmp_inputs, inferred_labels[base: base+batch_size], **learn_kwargs)
                elif algorithm.value == ContinuouLearningAlgorithm.ground_truth.value:
                    self.learn(tmp_inputs, tmp_true_labels, **learn_kwargs)
                elif algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                    self.learn_knowledge_distillation(
                        tmp_inputs, teacher_logits[base: base + batch_size],
                        tmp_true_labels, **learn_kwargs)
                else:
                    raise ValueError(f'Unknown algorithm {algorithm} {algorithm.value}')

            if self.drift_detector is not None:
                for j, (vt, vp) in enumerate(zip(tmp_true_labels, self.predict(tmp_inputs_client))):
                    self.drift_detector.update(int(not vt == vp))
                    if self.is_concept_drift():
                        drifts = np.append(drifts, len(y_preds) + j)
                        if concept_react_func is not None:
                            concept_react_func(self)

        if hasattr(self, 'drifts'):
            drifts = self.drifts
        return y_preds, ds.y.to_numpy(), ds._y.to_numpy(), drifts

    def is_concept_drift(self, *args, **kwargs):
        return self.drift_detector is not None and self.drift_detector.drift_detected


class EnhancedMlpClassifier(ClassifierMixin, BaseEnhancedMlp):
    """Enhanced version of sklearn classifier neural network, with pruning
    techniques and dropout implemented.
    """

    def __init__(
            self, classes: list[int],
            drift_detector=None, dropout: float = 0, prune_masks: list[ndarray] = None, hidden_layer_sizes=(100,),
            activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',
            learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4,
            verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        super().__init__(classes, drift_detector, dropout, prune_masks, hidden_layer_sizes, activation, solver=solver,
                         alpha=alpha, batch_size=batch_size, learning_rate_init=learning_rate_init, power_t=power_t,
                         loss='log_loss', max_iter=max_iter, shuffle=shuffle, learning_rate=learning_rate,
                         random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,
                         momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                         epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    @property
    def prunable(self):
        # keep the last layer
        return tuple(i for i in range(len(self.coefs_) - 1))

    def predict_proba(self, X: list, *args, log=False, as_dict, **kwargs) -> list[float]:
        try:
            r = MLPClassifier.predict_proba(self, X)
        except NotFittedError:
            r = np.zeros((len(X), len(self.classes)))
            r[:, 0] = 1
        if log:
            r = np.log(r, out=r)
        if as_dict:
            r = [dict(enumerate(v))for v in r]
        return r

    def predict(self, X: list, *args, **kwargs) -> list[int]:
        try:
            return MLPClassifier.predict(self, X)
        except (NotFittedError, AttributeError):
            return np.zeros(len(X))

    def predict_log_proba(self, X, *args, **kwargs):
        return self.predict_proba(X, *args, log=True, **kwargs)

    partial_fit = MLPClassifier.partial_fit
    _score = MLPClassifier._score
    _validate_input = MLPClassifier._validate_input
    _predict = MLPClassifier._predict
    _more_tags = MLPClassifier._more_tags

class EnhancedMlpRegressor(RegressorMixin, BaseEnhancedMlp):
    """Enhanced version of sklearn regressor neural network, with pruning
    techniques and dropout implemented.
    """

    def __init__(
            self, classes: list[int],
            drift_detector=None, dropout: float = 0, prune_masks: list[ndarray] = None, hidden_layer_sizes=(100,),
            activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',
            learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4,
            verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        super().__init__(classes, drift_detector, dropout, prune_masks, hidden_layer_sizes, activation, solver=solver,
                         alpha=alpha, batch_size=batch_size, learning_rate_init=learning_rate_init, power_t=power_t,
                         loss='squared_error', max_iter=max_iter, shuffle=shuffle, learning_rate=learning_rate,
                         random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,
                         momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                         epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    @property
    def prunable(self):
        # keep the last layer
        return tuple(i for i in range(len(self.coefs_)-1))

    def predict_proba(self, X: list, *, log=False, all_classes=False, as_dict=False, **kwargs) -> list[float]:
        if as_dict:
            all_classes = True
        try:
            if not hasattr(self, 'coefs_'):
                raise NotFittedError
            r = MLPRegressor.predict(self, X)
            if all_classes:
                ret = []
                for v in r:
                    tmp = np.zeros(len(self.classes))
                    floor = math.floor(v)
                    ceil = math.ceil(v)
                    if len(self.classes) == ceil:
                        tmp[floor-1] = v-floor
                        tmp[ceil-1] = ceil-v
                    else:
                        tmp[floor] = ceil-v
                        tmp[ceil] = v-floor
                    ret.append(tmp)
                r = np.array(ret)
        except NotFittedError:
            if all_classes:
                r = np.zeros((len(X), len(self.classes)))
                r[:, 0] = 1
            else:
                r = np.zeros((len(X)))
        if log:
            r = np.log(r, out=r)
        if as_dict:
            r = [dict(enumerate(v)) for v in r]
        return r

    def predict_proba_log(self, X, *args, **kwargs):
        return self.predict_proba(X, *args, log=True, **kwargs)

    def predict(self, X, *args, **kwargs) -> ndarray:
        try:
            if not hasattr(self, 'coefs_'):
                raise NotFittedError
            return MLPRegressor.predict(self, X).clip(min=np.min(self.classes), max=np.max(self.classes)).round()
        except (NotFittedError, AttributeError):
            return np.zeros(len(X))

    _predict = MLPRegressor._predict
    _score = MLPRegressor._score
    _validate_input = MLPRegressor._validate_input
    partial_fit = MLPRegressor.partial_fit
