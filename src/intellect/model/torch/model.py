"""
Module defining the model classed wrapping torch networks, providing further capabilities
to make them compliant with the INTELLECT pipeline.
"""
import inspect
import io
import os
import sys
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from river.base import DriftDetector
from sklearn.metrics import accuracy_score
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...dataset import ContinuouLearningAlgorithm, Dataset, InputForLearn
from ...scoring import compute_metric_percategory
from ..base import BaseModel


class TorchModel(BaseModel):
    """Wrapper class for torch models, providing them capabilities to be compliant
    with the INTELLECT prune/rank/learn pipeline.
    """
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs.get('drift_detector', None))
        self.init_params: dict[str, object] = kwargs
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.optimizer: torch.optim.AdamW = None
        self.loss_fn: torch.nn.CrossEntropyLoss = None

    @abstractmethod
    def _parse_ypred(self, y: torch.Tensor) -> np.ndarray:
        """Method to parse the predicted soft labels according to
        the type of the current model.

        Args:
            y (torch.Tensor): the predicted soft labels

        Returns:
            np.ndarray: the resulting value parsed.
        """

    @property
    def is_autoencoder(self) -> bool:
        """Method to return whether the current model is an autoencoder.

        Returns:
            bool: true if the current model is an autoencoder.
        """
        return len(self.init_params['in_features']) == self.init_params['n_outputs']

    @property
    def is_discrete(self) -> bool:
        """Method to return whether the current model is a discrete one (Regressor) or not

        Returns:
            bool: true if the current model is a regressor
        """
        return self.init_params['n_outputs'] == 1 and self.final_act not in ('Softmax', 'LogSoftmax')

    @property
    def device(self) -> str:
        """Method to return the type of device the model is currently running on.

        Returns:
            str: the name of the device
        """
        return next(self.model.parameters()).device.type

    @property
    def xtype(self) -> torch.dtype:
        """Method to return the dtype of the input data accepted by the network.

        Returns:
            torch.dtype: the dtype of the input data
        """
        return next(self.model.parameters()).dtype

    def cuda(self):
        """Method to move the current torch model instance into the gpu
        """
        self.model.cuda()
        self.optimizer_to_device('cuda')

    def cpu(self):
        """Method to move the current torch model instance into the cpu
        """
        self.model.cpu()
        self.optimizer_to_device('cpu')

    def optimizer_to_device(self, device: str):
        """Method to move the current optimizer, if any, into the same
        device the wrapped model is running on.

        Args:
            device (str): the target device.
        """
        if not self.optimizer:
            return
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
                continue
            if isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def copy_prune(self, other: 'TorchModel'):
        """Function to copy the prune mask from another given model into the current one.

        Args:
            other (TorchModel): the model to copy the prune mask from.
        """
        for l1, l2 in zip(self.model.children(), other.model.children()):
            for param in ('weight', 'bias'):
                if not hasattr(l2, f'{param}_mask'):
                    continue
                prune.custom_from_mask(l1, param, getattr(l2, f'{param}_mask').detach())

    def clone(self, init=True):
        if init is False:
            buffer = io.BytesIO()
            torch.save({'init_params': self.init_params, 'model': self.model,
                       'drift_detector': self.drift_detector}, buffer)
            buffer.seek(0)
            tmp = torch.load(buffer)
            ret = self.__class__()
            for k, v in tmp.items():
                setattr(ret, k, v)
            return ret
        new = self.__class__(**self.init_params)
        new.copy_prune(self)
        return new

    def save(self, path: str):
        if not path.endswith('.pt'):
            path += '.pt'
        with open(path, 'wb') as fp:
            torch.save({'init_params': self.init_params, 'model': self.model,
                       'drift_detector': self.drift_detector}, fp)

    @classmethod
    def load(cls, path: str) -> 'TorchModel':
        if not path.endswith('.pt'):
            path += '.pt'
        try:
            tmp = torch.load(path)
        except RuntimeError:
            tmp = torch.load(path, map_location='cpu')
        new = cls()
        for k, v in tmp.items():
            setattr(new, k, v)
        new.model.eval()
        return new

    def safe_cast_input(self, x: torch.Tensor | dict | pd.DataFrame | np.ndarray, is_y=False) -> torch.Tensor:
        """Function to correctly parse the given input into a compatible one for the neural network.

        Args:
            x (torch.Tensor | dict | pd.DataFrame | np.ndarray): the input data
            is_y (bool, optional): True if the provided data refers to classes. Defaults to False.

        Returns:
            torch.Tensor: the data correctly converted into tensor
        """
        ret: torch.Tensor = None
        dtype = torch.long if is_y and not self.is_autoencoder and not self.is_discrete else self.xtype

        if isinstance(x, torch.Tensor):
            ret = x
        elif isinstance(x, dict):
            ret = torch.tensor([0 if k not in x else x[k]
                               for k in self.init_params['in_features']])
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            ret = torch.from_numpy(x.to_numpy())
        elif isinstance(x, np.ndarray):
            ret = torch.from_numpy(x)
        else:
            return torch.tensor([x], dtype=dtype, device=self.device)
        if ret.ndim == 1 and not is_y:
            ret = ret[None, :]
        return ret.to(self.device, dtype=dtype)

    def predict(self, X: torch.Tensor, *args, **kwargs):
        return self._parse_ypred(self.predict_proba(X, *args, **kwargs))

    def predict_proba(self, X: torch.Tensor, *args, **kwargs):
        with torch.no_grad():
            y = self.model(self.safe_cast_input(X)).detach().cpu()
        if self.final_act == 'LogSoftmax':
            y = torch.exp(y)
        if self.is_discrete:
            y = y.squeeze()
        y = y.numpy()
        if kwargs.get('as_dict', False):
            return [{i: v.item() for i, v in enumerate(j)} for j in y]
        return y

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset | float = 0.2, batch_size: int = 256,
            max_epochs: int = 100, epochs_wo_improve: int = None, metric=accuracy_score, monitor_ds=None,
            shuffle=True, optimizer: torch.optim.Optimizer = None, loss_fn: torch.nn.modules.loss._Loss = None,
            higher_better=True, algorithm: ContinuouLearningAlgorithm = ContinuouLearningAlgorithm.ground_truth,
            learn_input=InputForLearn.client, oracle=None, optim_kwargs=None, learn_kwargs=None,
            idx_active_features=None, idx_active_features_oracle=None, concept_react_func=None):
        if optim_kwargs is None:
            optim_kwargs = {}
        if learn_kwargs is None:
            learn_kwargs = {}
        if idx_active_features is None:
            idx_active_features = []
        if idx_active_features_oracle is None:
            idx_active_features_oracle = []
        if metric is None:
            higher_better = False

        scaler = None
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.autograd.profiler.profile(enabled=False)
            torch.autograd.profiler.emit_nvtx(enabled=False)
            torch.autograd.set_detect_anomaly(mode=False)
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        monitored_metrics = []
        best_metric = 0 if higher_better else sys.maxsize
        current_epochs_wo_improve = 0
        history = {'loss_train': []}
        if metric is not None:
            history[f'{metric.__name__}_train'] = []

        best_state_dict = deepcopy(self.model.state_dict())
        self.check_optim_loss(optimizer, loss_fn, **optim_kwargs)

        val_loader = None
        if validation_dataset is not None and validation_dataset != 0:
            if isinstance(validation_dataset, float):
                validation_dataset = train_dataset.sample(validation_dataset, by_category=True)
                train_dataset = train_dataset.filter_indexes(validation_dataset.X.index.values)
            if metric is not None:
                history[f'{metric.__name__}_validation'] = []
            history['loss_validation'] = []
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=os.cpu_count(), pin_memory=True, persistent_workers=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=os.cpu_count(), pin_memory=True, persistent_workers=False)

        for i in (pbar := tqdm(range(max_epochs))):
            training_loss, training_pred, training_true = 0.0, None, None

            for _, (inputs, labels, _) in enumerate(train_loader):
                inputs_client = inputs_oracle = inputs
                true_labels = labels

                if idx_active_features:
                    inputs_client = inputs.detach().clone()
                    inputs_client[:, idx_active_features] = 0.

                if idx_active_features_oracle:
                    inputs_oracle = inputs.detach().clone()
                    inputs_oracle[:, idx_active_features_oracle] = 0.

                if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                    inferred_labels = oracle.predict(inputs_oracle)

                p = None
                if learn_input.value == InputForLearn.oracle.value:
                    inputs = inputs_oracle
                elif learn_input.value == InputForLearn.client.value:
                    inputs = inputs_client
                elif learn_input.value == InputForLearn.mixed.value:
                    p = np.random.permutation(inputs.shape[0]*2)
                    inputs = torch.concatenate((inputs_client, inputs_oracle))[p]
                    true_labels = true_labels.repeat(2)[p]
                    # the following two operations might be an overkill,
                    # think whether to perform only if ground_truth and knowledge_distill
                    inputs_oracle = inputs_oracle.repeat(2)[p]
                    inferred_labels = inferred_labels.repeat(2)[p]
                else:
                    raise ValueError(f'Unknown learn input {learn_input} {learn_input.value}')

                if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                    predictions, loss = self.learn(inputs, inferred_labels, scaler=scaler)
                elif algorithm.value == ContinuouLearningAlgorithm.ground_truth.value:
                    predictions, loss = self.learn(inputs, true_labels, scaler=scaler)
                elif algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                    predictions, loss = self.learn_knowledge_distillation(
                        oracle, inputs, inputs_oracle, true_labels, **learn_kwargs)
                else:
                    raise ValueError(f'Unknown algorithm {algorithm} {algorithm.value}')

                if p:
                    p = np.argsort(p)
                    predictions = np.split(predictions[p], 2)[0]
                    true_labels = np.split(true_labels[p], 2)[0]

                training_loss += loss
                training_pred = predictions if training_pred is None else np.concatenate(
                    (training_pred, predictions), axis=0)
                training_true = true_labels if training_true is None else np.concatenate(
                    (training_true, true_labels), axis=0)

                if self.drift_detector is not None:
                    for vt, vp in zip(labels, self.predict(inputs_client)):
                        self.drift_detector.update(int(not vt == vp))
                        if self.is_concept_drift():
                            if concept_react_func is not None:
                                concept_react_func(self)

            training_loss /= len(train_loader)
            metric_train = training_loss
            history['loss_train'].append(training_loss)
            if metric is not None:
                metric_train = metric(training_true, self._parse_ypred(training_pred))
                history[f'{metric.__name__}_train'].append(metric_train)
            eval_metric = metric_train

            if val_loader:
                val_loss, val_pred, val_true = 0, None, None
                with torch.no_grad():
                    for _, (a, b, _) in enumerate(val_loader):
                        y_raw_val = self.model(self.safe_cast_input(a)).detach()
                        if self.is_discrete:
                            y_raw_val = y_raw_val.squeeze()
                        val_loss += self.loss_fn(y_raw_val, self.safe_cast_input(b, is_y=True)).item()
                        val_true = b if val_true is None else np.concatenate((val_true, b), axis=0)
                        val_pred = y_raw_val.cpu() if val_pred is None else np.concatenate(
                            (val_pred, y_raw_val.cpu()), axis=0)
                val_loss /= len(val_loader)
                history['loss_validation'].append(val_loss)
                metric_validation = val_loss
                if metric is not None:
                    metric_validation = metric(val_true, self._parse_ypred(val_pred))
                    history[f'{metric.__name__}_validation'].append(metric_validation)
                eval_metric = metric_validation

            if monitor_ds:
                monitored_metrics.append(compute_metric_percategory(
                    monitor_ds.y, self.predict(monitor_ds.X),
                    monitor_ds._y, scorer=metric or accuracy_score))

            latest_param = dict((k, v[-1]['Global'] if isinstance(v[-1], dict) else v[-1])
                                for k, v in history.items())
            pbar.set_description(f'Epoch {i+1} {latest_param}', refresh=False)

            if not epochs_wo_improve:
                continue

            cond = eval_metric > best_metric if higher_better else eval_metric < best_metric
            current_epochs_wo_improve += 1
            if cond:
                current_epochs_wo_improve = 0
                best_metric = eval_metric
                best_state_dict = deepcopy(self.model.state_dict())

            if current_epochs_wo_improve == epochs_wo_improve:
                break
        self.model.load_state_dict(best_state_dict)
        if monitored_metrics:
            return history, monitored_metrics
        return history

    def continuous_learning(
            self, data: Dataset, algorithm: ContinuouLearningAlgorithm = ContinuouLearningAlgorithm.ground_truth,
            oracle: BaseModel = None, epochs: int = 1, batch_size: int = 'auto', idx_active_features: list[int] = None,
            idx_active_features_oracle: list[int] = None, learn_input: InputForLearn = InputForLearn.client,
            optimizer: torch.optim.Optimizer = None, loss_fn: torch.nn.modules.loss._Loss = None,
            optim_kwargs=None, learn_kwargs=None, concept_react_func=None):
        if optim_kwargs is None:
            optim_kwargs = {'weight_decay': 0.}
        if learn_kwargs is None:
            learn_kwargs = {}
        if idx_active_features is None:
            idx_active_features = []
        if idx_active_features_oracle is None:
            idx_active_features_oracle = []

        scaler = None
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.autograd.profiler.profile(enabled=False)
            torch.autograd.profiler.emit_nvtx(enabled=False)
            torch.autograd.set_detect_anomaly(mode=False)
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.check_optim_loss(optimizer, loss_fn, **optim_kwargs)

        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            num_workers=os.cpu_count(), pin_memory=True, persistent_workers=False)

        y_preds, y_true, y_labels, drifts = np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        for (inputs, labels, type_labels) in tqdm(loader):
            inputs_client = inputs_oracle = inputs
            true_labels = labels
            type_labels = np.array(type_labels)

            if idx_active_features:
                inputs_client = inputs.detach().clone()
                inputs_client[:, idx_active_features] = 0.

            if idx_active_features_oracle:
                inputs_oracle = inputs.detach().clone()
                inputs_oracle[:, idx_active_features_oracle] = 0.

            if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                inferred_labels = oracle.predict(inputs_oracle)

            if learn_input.value == InputForLearn.oracle.value:
                inputs = inputs_oracle
            elif learn_input.value == InputForLearn.client.value:
                inputs = inputs_client
            elif learn_input.value == InputForLearn.mixed.value:
                inputs = torch.concatenate((inputs_client, inputs_oracle))
                inferred_labels = inferred_labels.repeat(2)
                true_labels = true_labels.repeat(2)
                inputs_oracle = inputs_oracle.repeat(2)
            else:
                raise ValueError(f'Unknown learn input {learn_input} {learn_input.value}')

            for i in range(epochs):
                p = np.random.permutation(len(true_labels))

                if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                    predictions, _ = self.learn(inputs[p], inferred_labels[p], scaler=scaler)
                elif algorithm.value == ContinuouLearningAlgorithm.ground_truth.value:
                    predictions, _ = self.learn(inputs[p], true_labels[p], scaler=scaler)
                elif algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                    predictions, _ = self.learn_knowledge_distillation(
                        oracle, inputs[p], inputs_oracle[p], true_labels[p], **learn_kwargs)
                else:
                    raise ValueError(f'Unknown algorithm {algorithm} {algorithm.value}')

                if i != 0:
                    continue

                if learn_input.value == InputForLearn.mixed.value:
                    p = np.argsort(p)
                    predictions = np.split(predictions[p], 2)[0]

                y_true = np.concatenate((y_true, labels), axis=0)
                y_labels = np.concatenate((y_labels, type_labels), axis=0)
                y_preds = np.concatenate((y_preds, self._parse_ypred(predictions)), axis=0)

            if self.drift_detector is not None:
                for j, (vt, vp) in enumerate(zip(labels, self.predict(inputs_client))):
                    self.drift_detector.update(int(not vt == vp))
                    if self.is_concept_drift():
                        drifts = np.append(drifts, len(y_preds) + j)
                        if concept_react_func is not None:
                            concept_react_func(self)

        return y_preds, y_true, y_labels, drifts

    @property
    def prunable(self) -> list[torch.nn.Module]:
        """Property to return the list of prunable modules within the model

        Returns:
            list[torch.nn.Module]: list of prunable modules
        """
        return [m for m in self.model.children() if isinstance(m, torch.nn.Linear)][:-1]

    def learn_knowledge_distillation(
            self, oracle: BaseModel, inputs: torch.Tensor, inputs_oracle: torch.Tensor, true_labels: torch.Tensor,
            temperature: int = 1, alpha: float = 1.) -> tuple[torch.Tensor, float]:
        """Function to perform a step of training using the knowledge distillation algorithm
        with the provided arguments.

        Args:
            oracle (BaseModel): the teacher model.
            inputs (torch.Tensor): the input for the current student model.
            inputs_oracle (torch.Tensor): the input for the teacher model.
            true_labels (torch.Tensor): the true labels of the input data.
            temperature (int, optional): used to soften the probability distribution of the teacher model's
                predictions, making them more informative for the student model. Defaults to 1.
            alpha (float, optional): weight assigned to the knowledge loss component in the overall
                loss function, controlling the influence of the distilled knowledge on the student
                model's training. Alpha=1 ignores the true labels, alpha=0 considers only true
                labels. Defaults to 1.

        Returns:
            tuple[torch.Tensor, float]: _description_
        """
        if self.optimizer is None or self.loss_fn is None:
            self.check_optim_loss(self.optimizer, self.loss_fn)

        inputs = self.safe_cast_input(inputs)
        teacher_logits = self.safe_cast_input(oracle.predict_proba(inputs_oracle), is_y=True)
        print(teacher_logits)
        self.model.train()
        self.optimizer.zero_grad()
        predicted = self.model(inputs)
        if self.is_discrete:
            predicted = predicted.squeeze()
        target_loss = self.loss_fn(predicted, true_labels)

        if self.final_act == 'LogSoftmax':
            predicted = torch.exp(predicted)
        a = log_softmax(predicted / temperature, dim=-1)
        b = softmax(teacher_logits / temperature, dim=-1)
        distill_loss = KLDivLoss(reduction='batchmean')(a, b) * (temperature**2)

        loss = alpha * distill_loss + (1 - alpha) * target_loss
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return predicted.detach().cpu(), loss.item()

    def learn(self, X: torch.Tensor, y: torch.Tensor, scaler: torch.cuda.amp.GradScaler = None):
        if self.optimizer is None or self.loss_fn is None:
            self.check_optim_loss(self.optimizer, self.loss_fn)
        self.model.train()
        X, y = self.safe_cast_input(X), self.safe_cast_input(y, is_y=True)
        self.optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = self.model(X)
                if self.is_discrete:
                    outputs = outputs.squeeze()
                loss: torch.Tensor = self.loss_fn(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            outputs = self.model(X)
            if self.is_discrete:
                outputs = outputs.squeeze()
            loss: torch.Tensor = self.loss_fn(outputs, y)
            loss.backward()
            self.optimizer.step()
        self.model.eval()
        return outputs.detach().cpu(), loss.item()

    @property
    def final_act(self) -> str:
        """Property to return the name of the output activation function.

        Returns:
            str: the name of the output activation function class.
        """
        return self.model[-1].__class__.__name__

    def check_optim_loss(self, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss, **kwargs):
        """Method to set the provided optimizers and loss functions or the default ones.

        Args:
            optimizer (torch.optim.Optimizer): optimizer to be set, or None to autoselect.
                Additional kwargs are passed to the optimizer initialization function.
            loss_fn (torch.nn.modules.loss._Loss): loss function to be set, or None to autoselect.

        Raises:
            ValueError: when not matching the output layer activation.
        """
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)
        if loss_fn is None:
            if self.is_discrete:
                loss_fn = torch.nn.BCELoss()
            elif self.final_act == 'LogSoftmax':
                loss_fn = torch.nn.NLLLoss()
            elif self.final_act == 'Softmax':
                loss_fn = torch.nn.CrossEntropyLoss()
            elif self.final_act in ('Identity', 'Sigmoid'):
                loss_fn = torch.nn.MSELoss()
            else:
                raise ValueError(f'Unknown {self.final_act}')
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def is_concept_drift(self, *args, **kwargs):
        return self.drift_detector is not None and self.drift_detector.drift_detected


class Mlp(TorchModel):
    """Multilayer perceptron implementation with dropout and pruning ranking capabilities.

    Args:
        in_features (list[str], optional): list of input feature names. Defaults to None.
        n_outputs (int, optional): number of outputs (depend by problem type). Defaults to 2.
        hidden_units (int, optional): number of units in each hidden layer. Defaults to None.
        hidden_layers (int, optional): number of hidden layers. Defaults to None.
        batch_norm (bool, optional): whether to use batch normalization before
            each linear layer. Defaults to False.
        dropout_hidden (float, optional): dropout ratio for hidden layers. Defaults to 0..
        dropout_input (float, optional): dropout ratio for input layer. Defaults to 0..
        activation (str, optional): activation function of each linear layer. Defaults to 'ReLU'.
        final_activation (str, optional): output activation of the model. Defaults to 'LogSoftmax'.
        dtype (torch.dtype, optional): data type (for precision). Defaults to torch.float.
        drift_detector (DriftDetector, optional): drift detector object. Defaults to None.
    """

    def __init__(
            self, in_features: list[str] = None, n_outputs: int = 2,
            hidden_units: int = None, hidden_layers: int = None, batch_norm: bool = False,
            dropout_hidden: float = 0., dropout_input: float = 0., activation: str = 'ReLU',
            final_activation: str = 'LogSoftmax', dtype: torch.dtype = torch.float,
            drift_detector: DriftDetector = None):
        super().__init__(in_features=in_features, n_outputs=n_outputs, hidden_units=hidden_units,
                         batch_norm=batch_norm, dropout_hidden=dropout_hidden, dropout_input=dropout_input,
                         hidden_layers=hidden_layers, activation=activation,
                         final_activation=final_activation, dtype=dtype, drift_detector=drift_detector)

        if hidden_layers is None:
            return

        activation: torch.nn.ReLU = getattr(torch.nn, activation)
        final_activation: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)
        final_kwargs = {'dim': -1} if 'dim' in inspect.signature(final_activation.__init__).parameters else {}

        if dropout_input > 0.:
            self.model.add_module('input_drop', torch.nn.Dropout(p=dropout_input))

        previous = len(in_features)
        for i in range(hidden_layers+1):
            name = 'input' if i == 0 else 'hidden'
            if batch_norm:
                self.model.add_module(f'{name}_{i}_batch', torch.nn.BatchNorm1d(previous, dtype=dtype))
            self.model.add_module(f'{name}_{i}_layer', torch.nn.Linear(
                previous, hidden_units, dtype=dtype))
            self.model.add_module(f'{name}_{i}_act', activation())
            if dropout_hidden > 0.:
                self.model.add_module(f'{name}_{i}_drop', torch.nn.Dropout(p=dropout_hidden))
            previous = hidden_units

        self.model.add_module('final_layer', torch.nn.Linear(previous, n_outputs, dtype=dtype))
        self.model.add_module('final_act', final_activation(**final_kwargs))

    def _parse_ypred(self, y: torch.Tensor):
        if self.is_discrete:
            return y.squeeze().round()
        if y.ndim > 1:
            return np.argmax(y, axis=-1)
        return y


class MlpEncoder(TorchModel):
    """Multilayer perceptron encoder with dropout and pruning capabilities.

    Args:
        in_features (list[str], optional): list of input features. Defaults to None.
        hidden_units (int, optional): number of units in each hidden layer. Defaults to None.
        hidden_layers (int, optional): number of hidden layers. Defaults to None.
        dropout_hidden (float, optional): dropout ratio in hidden layers. Defaults to 0..
        dropout_input (float, optional): dropout ratio in input layer. Defaults to 0..
        activation (str, optional): activation function after each linear layer. Defaults to 'ReLU'.
        final_activation (str, optional): output activation function. Defaults to 'Sigmoid'.
        dtype (torch.dtype, optional): data type to match (for precision). Defaults to torch.float.
        drift_detector (DriftDetector, optional): optional drift detector object. Defaults to None.
    """

    def __init__(
            self, in_features: list[str] = None, hidden_layers: int = None,
            dropout_hidden: float = 0., dropout_input: float = 0., activation: str = 'ReLU',
            final_activation: str = 'Sigmoid', dtype: torch.dtype = torch.float,
            drift_detector: DriftDetector = None):
        super().__init__(in_features=in_features, n_outputs=len(in_features), dropout_hidden=dropout_hidden,
                         dropout_input=dropout_input, hidden_layers=hidden_layers, activation=activation,
                         final_activation=final_activation, dtype=dtype, drift_detector=drift_detector)

        if hidden_layers is None:
            return

        activation: torch.nn.ReLU = getattr(torch.nn, activation)
        final_activation: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)

        if dropout_input > 0.:
            self.model.add_module('input_drop', torch.nn.Dropout(p=dropout_input))

        current_size, next_size = len(in_features), 0
        for i in range(hidden_layers):
            name = 'input' if i == 0 else 'encoder'
            next_size = round(current_size / 2)
            self.model.add_module(f'{name}_{i}_layer', torch.nn.Linear(
                current_size, next_size, dtype=dtype))
            self.model.add_module(f'{name}_{i}_act', activation())
            if dropout_hidden > 0.:
                self.model.add_module(f'{name}_{i}_drop', torch.nn.Dropout(p=dropout_hidden))
            current_size = next_size

        for i in range(hidden_layers):
            name = 'output' if i + 1 == hidden_layers else 'decoder'
            next_size = current_size * 2
            self.model.add_module(f'{name}_{i}_layer', torch.nn.Linear(
                current_size, next_size, dtype=dtype))
            if name == 'output':
                self.model.add_module('final_act', final_activation())
                break
            self.model.add_module(f'{name}_{i}_act', activation())
            if dropout_hidden > 0.:
                self.model.add_module(f'{name}_{i}_drop', torch.nn.Dropout(p=dropout_hidden))
            current_size = next_size

    def _parse_ypred(self, y):
        return y
