
"""
CRAFT Module for pytorch
"""

from abc import ABC, abstractmethod
from math import ceil
from typing import Callable, Optional

import torch
import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import NotFittedError

from .sobol.sampler import HaltonSequence
from .sobol.estimators import JansenEstimator


def torch_to_numpy(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return np.array(tensor)


def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda'):
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i*batch_size for i in range(nb_batchs)]

    results = []

    with torch.no_grad():
        for i in start_ids:
            x = torch.tensor(dataset[i:i+batch_size])
            x = x.to(device)

        if resize:
            x = torch.nn.functional.interpolate(
                x, size=resize, mode='bilinear', align_corners=False)

        results.append(model(x).cpu())

    results = torch.cat(results)
    return results


class BaseConceptExtractor(ABC):
    """
    Base class for concept extraction models.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.

    """

    def __init__(self, input_to_latent: Callable,
                 latent_to_logit: Optional[Callable] = None,
                 number_of_concepts: int = 20,
                 batch_size: int = 64):

        # sanity checks
        assert (number_of_concepts >
                0), "number_of_concepts must be greater than 0"
        assert (batch_size > 0), "batch_size must be greater than 0"
        assert (callable(input_to_latent)
                ), "input_to_latent must be a callable function"

        self.input_to_latent = input_to_latent
        self.latent_to_logit = latent_to_logit
        self.number_of_concepts = number_of_concepts
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, inputs):
        """
        Fit the CAVs to the input data.

        Parameters
        ----------
        inputs : array-like
            The input data to fit the model on.

        Returns
        -------
        tuple
            A tuple containing the input data and the matrices (U, W) that factorize the data.

        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs):
        """
        Transform the input data into a concepts embedding.

        Parameters
        ----------
        inputs : array-like
            The input data to transform.

        Returns
        -------
        array-like
            The transformed embedding of the input data.

        """
        raise NotImplementedError


class Craft(BaseConceptExtractor):
    """
    Class Implementing the CRAFT Concept Extraction Mechanism.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable, optional
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.
    patch_size : int, optional
        The size of the patches to extract from the input data. Default is 64.
    """

    def __init__(self, input_to_latent: Callable,
                 latent_to_logit: Optional[Callable] = None,
                 number_of_concepts: int = 20,
                 batch_size: int = 64,
                 patch_size: int = 64,
                 device: str = 'cuda'):
        super().__init__(input_to_latent, latent_to_logit, number_of_concepts, batch_size)

        self.patch_size = patch_size
        self.activation_shape = None
        self.device = device

    def fit(self, inputs):
        """
        Fit the Craft model to `inputs`.

        Parameters
        ----------
        inputs: 4-D tensor | 2-D array | list/tuple
        * Images  .......... (N, C, H, W)   — comportement historique.
        * Activations 2-D .. (N, D)         — déjà calculées, positives.
            * Liste/tuple ....... données brutes(ex. list[str])   —
            seront encodées via `input_to_latent`.

        Returns
        -------
        (patches, U, W)
        patches: torch.Tensor | None
            Les patches extraits pour les images, sinon None.
            U: np.ndarray
            Les activations conceptuelles (N, R).
            W: np.ndarray
            La base de concepts (R, D).
        """
        # Datatype to analyse : image, no changes
        # Add text format
        is_image = isinstance(inputs, torch.Tensor) and inputs.ndim == 4
        is_act2d = isinstance(
            inputs, (np.ndarray, torch.Tensor)) and inputs.ndim == 2
        is_sequence = isinstance(inputs, (list, tuple))

        if not (is_image or is_act2d or is_sequence):
            raise ValueError("Unsupported input type for Craft.fit")

        # If image, same behaviour
        if is_image:
            assert len(
                inputs.shape) == 4, "Input data must be of shape (n_samples, channels, height, width)."
            assert inputs.shape[2] == inputs.shape[3], "Input data must be square."

            image_size = inputs.shape[2]

            # extract patches from the input data, keep patches on cpu
            strides = int(self.patch_size * 0.80)

            patches = torch.nn.functional.unfold(
                inputs, kernel_size=self.patch_size, stride=strides)
            patches = patches.transpose(1, 2).contiguous(
            ).view(-1, 3, self.patch_size, self.patch_size)

            # encode the patches and obtain the activations
            activations = _batch_inference(self.input_to_latent, patches, self.batch_size, image_size,
                                           device=self.device)

            # if the activations have shape (n_samples, height, width, n_channels),
            # apply average pooling
            if len(activations.shape) == 4:
                activations = torch.mean(activations, dim=(2, 3))

        # ACtivations 2d
        elif is_act2d:
            patches = None
            activations = torch.as_tensor(inputs, device=self.device)

        # List / Tupple  (ex. text)
        else:
            patches = None
            batches = []
            # we don't use _batch_inference to not change it
            for i in range(0, len(inputs), self.batch_size):
                # sous-liste de chaînes
                sub_texts = inputs[i: i + self.batch_size]
                with torch.no_grad():
                    acts = self.input_to_latent(
                        sub_texts).to(self.device)  # (b, D)
                batches.append(acts)
            activations = torch.cat(batches, dim=0)              # (N, D)

        assert torch.min(activations) >= 0.0, "Activations must be positive."

        # apply NMF to the activations to obtain matrices U and W
        reducer = NMF(n_components=self.number_of_concepts)
        U = reducer.fit_transform(torch_to_numpy(activations))
        W = reducer.components_.astype(np.float32)

        # store the factorizer and W as attributes of the Craft instance
        self.reducer = reducer
        self.W = np.array(W, dtype=np.float32)

        return patches, U, W

    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """

        if not hasattr(self, 'reducer'):
            raise NotFittedError(
                "The factorization model has not been fitted to input data yet.")

    def transform(self, inputs: np.ndarray, activations: Optional[np.ndarray] = None):
        self.check_if_fitted()

        if activations is None:
            activations = _batch_inference(self.input_to_latent, inputs, self.batch_size,
                                           device=self.device)

        is_4d = len(activations.shape) == 4

        if is_4d:
            # (N, C, W, H) -> (N * W * H, C)
            activation_size = activations.shape[-1]
            activations = activations.permute(0, 2, 3, 1)
            activations = torch.reshape(
                activations, (-1, activations.shape[-1]))

        W_dtype = self.reducer.components_.dtype
        U = self.reducer.transform(torch_to_numpy(activations).astype(W_dtype))

        if is_4d:
            # (N * W * H, R) -> (N, W, H, R)
            U = np.reshape(
                U, (-1, activation_size, activation_size, U.shape[-1]))

        return U

    def estimate_importance(self, inputs, class_id, nb_design=32):
        """
        Estimates the importance of each concept for a given class.

        Parameters
        ----------
        inputs : numpy array or Tensor
            The input data to be transformed.
        class_id : int
            The class id to estimate the importance for.
        nb_design : int, optional
            The number of design to use for the importance estimation. Default is 32.

        Returns
        -------
        importances : list
            The Sobol total index (importance score) for each concept.

        """
        self.check_if_fitted()

        U = self.transform(inputs)

        masks = HaltonSequence()(self.number_of_concepts,
                                 nb_design=nb_design).astype(np.float32)
        estimator = JansenEstimator()

        importances = []

        if len(U.shape) == 2:
            # apply the original method of the paper

            for u in U:
                u_perturbated = u[None, :] * masks
                a_perturbated = u_perturbated @ self.W

                y_pred = _batch_inference(self.latent_to_logit, a_perturbated, self.batch_size,
                                          device=self.device)
                y_pred = y_pred[:, class_id]

                stis = estimator(torch_to_numpy(masks),
                                 torch_to_numpy(y_pred),
                                 nb_design)

                importances.append(stis)

        elif len(U.shape) == 4:
            # apply a re-parameterization trick and use mask on all localization for a given
            # concept id to estimate sobol indices
            for u in U:
                u_perturbated = u[None, :] * masks[:, None, None, :]
                a_perturbated = np.reshape(
                    u_perturbated, (-1, u.shape[-1])) @ self.W
                a_perturbated = np.reshape(
                    a_perturbated, (len(masks), U.shape[1], U.shape[2], -1))
                a_perturbated = np.moveaxis(a_perturbated, -1, 1)

                y_pred = _batch_inference(self.latent_to_logit, a_perturbated, self.batch_size,
                                          device=self.device)
                y_pred = y_pred[:, class_id]

                stis = estimator(torch_to_numpy(masks),
                                 torch_to_numpy(y_pred),
                                 nb_design)

                importances.append(stis)

        return np.mean(importances, 0)

    def _factorize(self, activations):
        if activations.ndim == 4:          # (N, C, H, W) → pool
            activations = torch.mean(activations, (2, 3))
        assert torch.min(activations) >= 0, "Activations must be positive."
        reducer = NMF(n_components=self.number_of_concepts)
        U = reducer.fit_transform(torch_to_numpy(activations))
        self.reducer = reducer
        self.W = reducer.components_.astype(np.float32)
        return U
