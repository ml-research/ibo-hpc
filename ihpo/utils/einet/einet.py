import logging
from ihpo.utils.einet.utils import invert_permutation
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Union, Optional

import numpy as np
import torch
from torch import nn

from ihpo.utils.einet.layers.distributions.abstract_leaf import AbstractLeaf
from ihpo.utils.einet.layers.einsum import (
    EinsumLayer,
)
from ihpo.utils.einet.layers.mixing import MixingLayer
from ihpo.utils.einet.layers.factorized_leaf import FactorizedLeaf, FactorizedLeafSimple
from ihpo.utils.einet.layers.linsum import LinsumLayer, LinsumLayer2
from ihpo.utils.einet.layers.product import RootProductLayer
from ihpo.utils.einet.sampling_utils import index_one_hot, sampling_context, SamplingContext
from ihpo.utils.einet.layers.sum import SumLayer
from ihpo.utils.einet.type_checks import check_valid
from .dist import Domain, DataType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EinetConfig:
    """Class for the configuration of an Einet."""

    num_features: int = None  # Number of input features
    num_channels: int = 1  # Number of data input channels per feature
    num_sums: int = 10  # Number of sum nodes at each layer
    num_leaves: int = 10  # Number of distributions for each scope at the leaf layer
    num_repetitions: int = 5  # Number of repetitions
    num_classes: int = 1  # Number of root heads / Number of classes
    depth: int = 1  # Tree depth
    dropout: float = 0.0  # Dropout probabilities for leaves and sum layers
    leaf_type: Type = None  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_kwargs: Dict[str, Any] = field(default_factory=dict)  # Parameters for the leaf base class
    layer_type: str = "linsum"  # Indicates the intermediate layer type: linsum or einsum
    structure: str = "top-down"  # Structure of the Einet: top-down or bottom-up

    def assert_valid(self):
        """Check whether the configuration is valid."""

        # Check that each dimension is valid
        check_valid(self.depth, int, 0)
        check_valid(self.num_features, int, 2)
        check_valid(self.num_channels, int, 1)
        check_valid(self.num_classes, int, 1)
        check_valid(self.num_sums, int, 1)
        check_valid(self.num_repetitions, int, 1)
        check_valid(self.num_leaves, int, 1)
        check_valid(self.dropout, float, 0.0, 1.0, allow_none=True)
        assert self.leaf_type is not None, "EinetConfig.leaf_type parameter was not set!"
        assert self.layer_type in [
            "linsum",
            "linsum2",
            "einsum",
        ], f"Invalid layer type {self.layer_type}. Must be 'linsum' or 'einsum'."
        assert self.structure in [
            "top-down",
            "bottom-up",
        ], f"Invalid structure type {self.structure}. Must be 'top-down' or 'bottom-up'."

        assert isinstance(self.leaf_type, type) and issubclass(
            self.leaf_type, AbstractLeaf
        ), f"Parameter EinetConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_type}."

        # If the leaf layer is multivariate distribution, extract its cardinality
        if "cardinality" in self.leaf_kwargs:
            cardinality = self.leaf_kwargs["cardinality"]
        else:
            cardinality = 1

        if self.structure == "bottom-up":
            assert self.layer_type == "linsum", "Bottom-up structure only supports LinsumLayer due to handling of padding (not implemented for einsumlayer yet)."

        # Get minimum number of features present at the lowest layer (num_features is the actual input dimension,
        # cardinality in multivariate distributions reduces this dimension since it merges groups of size #cardinality)
        min_num_features = np.ceil(self.num_features // cardinality)
        assert (
            2**self.depth <= min_num_features
        ), f"The tree depth D={self.depth} must be <= {np.floor(np.log2(min_num_features))} (log2(in_features // cardinality))."


class Einet(nn.Module):
    """
    Einet RAT SPN PyTorch implementation with layer-wise tensors.

    See also:
    - RAT SPN: https://arxiv.org/abs/1806.01910
    - EinsumNetworks: https://arxiv.org/abs/2004.06231
    """

    def __init__(self, config: EinetConfig, domains: List[Domain]):
        """
        Create an Einet based on a configuration object.

        Args:
            config (EinetConfig): Einet configuration object.
        """
        super().__init__()
        config.assert_valid()
        self.config = config

        # Construct the architecture
        if self.config.structure == "top-down":
            self._build_structure_top_down()
        elif self.config.structure == "bottom-up":
            self._build_structure_bottom_up()
        else:
            raise ValueError(f"Invalid structure type {self.config.structure}. Must be '_riginal' or 'bottom-up'.")

        # Leaf cache
        self._leaf_cache = {}
        self.domains = domains
        self.cont_shift_parameters = None
        self.disc_shift_parameters = None

    @classmethod
    def get_hyperparameters(cls, num_feats: int, num_samples: int):
        """Return hyperparameters of Einet based on basic dataset statistics

        Args:
            num_feats (int): Number of features
            num_samples (int): Number of samples
        """
        return {
            'num_sums': int(np.round(5 + (1 / (np.log10(num_samples)*2*np.sqrt(num_samples))) * num_samples)),
            'num_repitions': int(np.round((1 / (np.log10(num_samples)*np.sqrt(num_samples))) * num_samples)),
            'num_leaves': int(np.round(((1 / (np.log10(num_samples)*0.8*np.sqrt(num_samples))) * num_samples))),
            'depth': int(np.round(np.log(num_feats))),
        }

    def init_shift_parameters(self, val=0):
        self.disc_shift_parameters = []
        num_cont_params = 0
        for dom in self.domains:
            if dom.data_type == DataType.DISCRETE:
                param_size = 2*(dom.max - dom.min) + 1
                param_vec = nn.Parameter(torch.zeros(param_size))
                self.disc_shift_parameters.append(param_vec)
            else:
                num_cont_params += 1
        self.cont_shift_parameters = nn.Parameter(torch.zeros((self.config.num_channels, num_cont_params)) + val)
        #if self.config.leaf_type.__name__ == 'PiecewiseLinear':
        #    self.leaf.base_leaf.init_shift_parameters()


    def reset_cache(self):
        """Reset the leaf cache."""
        self._leaf_cache = {}

    def forward(self, x: torch.Tensor, marginalized_scopes: torch.Tensor = None, cache_index: Optional[int] = None) -> torch.Tensor:
        """
        Inference pass for the Einet model.

        Args:
            x (torch.Tensor): Input data of shape [N, C, D], where C is the number of input channels (useful for images) and D is the number of features/random variables (H*W for images).
            marginalized_scopes (torch.Tensor):  (Default value = None)
            cache_index (Optional[int]): Index of the cache. If not None, the leaf tries to retrieve the cached log-likelihoods or computes the log-likelihoods on a cache-miss and then caches the results. (Default value = None)

        Returns:
            Log-likelihood tensor of the input: p(X) or p(X | C) if number of classes > 1.
        """

        # Add channel dimension if not present
        if x.dim() == 2:  # [N, D]
            x = x.unsqueeze(1)

        if x.dim() == 4:  # [N, C, H, W]
            x = x.view(x.shape[0], self.config.num_channels, -1)

        assert x.dim() == 3
        assert (
            x.shape[1] == self.config.num_channels
        ), f"Number of channels in input ({x.shape[1]}) does not match number of channels specified in config ({self.config.num_channels})."
        assert (
            x.shape[2] == self.config.num_features
        ), f"Number of features in input ({x.shape[0]}) does not match number of features specified in config ({self.config.num_features})."

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        # If cache_index is set, try to retrieve the cached leaf log-likelihoods
        if cache_index is not None and cache_index in self._leaf_cache:
            x = self._leaf_cache[cache_index]
        else:
            if self.cont_shift_parameters is not None:
                x = self.leaf(x, marginalized_scopes, cont_shift_params=self.cont_shift_parameters, disc_shift_params=self.disc_shift_parameters)
            else:
                x = self.leaf(x, marginalized_scopes)

            if cache_index is not None:  # Cache index was specified but not found in cache
                self._leaf_cache[cache_index] = x


        # Factorize input channels
        if not isinstance(self.leaf, (FactorizedLeaf, FactorizedLeafSimple)):
            x = x.sum(dim=1)
            assert x.shape == (
                x.shape[0],
                self.config.num_features,
                self.config.num_leaves,
                self.config.num_repetitions,
            ), f"Invalid shape after leaf layer. Was {x.shape} but expected ({x.shape[0]}, {self.config.num_features}, {self.config.num_leaves}, {self.config.num_repetitions})."
        else:
            assert x.shape == (
                x.shape[0],
                self.leaf.num_features_out,
                self.config.num_leaves,
                self.config.num_repetitions,
            ), f"Invalid shape after leaf layer. Was {x.shape} but expected ({x.shape[0]}, {self.leaf.num_features_out}, {self.config.num_leaves}, {self.config.num_repetitions})."

        # Apply permutation
        if hasattr(self, "permutation"):
            for i in range(self.config.num_repetitions):
                x[:, :, :, i] = x[:, self.permutation[i], :, i]

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        batch_size, features, channels, repetitions = x.size()
        assert features == 1  # number of features should be 1 at this point
        assert channels == self.config.num_classes

        # If model has multiple reptitions, perform repetition mixing
        if self.config.num_repetitions > 1:
            # Mix repetitions
            x = self.mixing(x)
        else:
            # Remove repetition index
            x = x.squeeze(-1)

        # Remove feature dimension
        x = x.squeeze(1)

        # Final shape check
        assert x.shape == (batch_size, self.config.num_classes)

        return x

    def _forward_layers(self, x):
        """
        Forward pass through the inner sum and product layers.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Output of the last layer before the root layer.
        """
        # Forward to inner product and sum layers
        for layer in self.layers:
            x = layer(x)
        return x

    def posterior(self, x) -> torch.Tensor:
        """
        Compute the posterior probability logp(y | x) of the data.

        Args:
          x: Data input.

        Returns:
            Posterior logp(y | x).
        """
        assert self.config.num_classes > 1, "Cannot compute posterior without classes."

        # logp(x | y)
        ll_x_g_y = self(x)  # [N, C]

        return posterior(ll_x_g_y, self.config.num_classes)

    def _build_structure_top_down(self):
        """Construct the internal architecture of the Einet."""
        # Build the SPN top down:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        intermediate_layers: List[Union[EinsumLayer, LinsumLayer, LinsumLayer2]] = []

        # Construct layers from top to bottom
        for i in np.arange(start=1, stop=self.config.depth + 1):
            # Choose number of input sum nodes
            # - if this is an intermediate layer, use the number of sum nodes from the previous layer
            # - if this is the first layer, use the number of leaves as the leaf layer is below the first sum layer
            if i < self.config.depth:
                _num_sums_in = self.config.num_sums
            else:
                _num_sums_in = self.config.num_leaves

            # Choose number of output sum nodes
            # - if this is the last layer, use the number of classes
            # - otherwise use the number of sum nodes from the next layer
            if i == 1:
                _num_sums_out = self.config.num_classes
            else:
                _num_sums_out = self.config.num_sums

            # Calculate number of input features: since we represent a binary tree, each layer merges two partitions,
            # hence viewing this from top down we have 2**i input features at the i-th layer
            in_features = 2**i

            if self.config.layer_type == "einsum":
                layer = EinsumLayer(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            elif self.config.layer_type == "linsum":
                layer = LinsumLayer(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            elif self.config.layer_type == "linsum2":
                layer = LinsumLayer2(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            else:
                raise ValueError(f"Unknown layer type {self.config.layer_type}")

            intermediate_layers.append(layer)

        if self.config.depth == 0:
            # Create a single sum layer
            layer = SumLayer(
                num_sums_in=self.config.num_leaves,
                num_features=1,
                num_sums_out=self.config.num_classes,
                num_repetitions=self.config.num_repetitions,
                dropout=self.config.dropout,
            )
            intermediate_layers.append(layer)

        # Construct leaf
        leaf_num_features_out = intermediate_layers[-1].num_features
        self.leaf = self._build_input_distribution(num_features_out=leaf_num_features_out)

        # List layers in a bottom-to-top fashion
        self.layers: List[Union[EinsumLayer, LinsumLayer, LinsumLayer2]] = nn.ModuleList(reversed(intermediate_layers))

        # If model has multiple reptitions, add repetition mixing layer
        if self.config.num_repetitions > 1:
            self.mixing = MixingLayer(
                num_features=1,
                num_sums_in=self.config.num_repetitions,
                num_sums_out=self.config.num_classes,
                dropout=self.config.dropout,
            )

        # Construct sampling root with weights according to priors for sampling
        if self.config.num_classes > 1:
            self._class_sampling_root = SumLayer(
                num_sums_in=self.config.num_classes,
                num_features=1,
                num_sums_out=1,
                num_repetitions=1,
            )
            self._class_sampling_root.weights = nn.Parameter(
                torch.log(
                    torch.ones(size=(1, self.config.num_classes, 1, 1)) * torch.tensor(1 / self.config.num_classes)
                ),
                requires_grad=False,
            )

    def _build_structure_bottom_up(self):
        """Construct the internal architecture of the Einet."""
        # Build the SPN bottom up:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        intermediate_layers: List[Union[EinsumLayer, LinsumLayer, LinsumLayer2]] = []

        # Construct layers from bottom to top
        in_features = self.config.num_features
        for i in np.arange(start=0, stop=self.config.depth):
            # Choose number of input sum nodes
            # - if this is an intermediate layer, use the number of sum nodes from the previous layer
            # - if this is the first layer, use the number of leaves as the leaf layer is below the first sum layer
            if i == 0:
                _num_sums_in = self.config.num_leaves
            else:
                _num_sums_in = self.config.num_sums

            # Choose number of output sum nodes
            # - if this is the last layer, use the number of classes
            # - otherwise use the number of sum nodes from the next layer

            # if i == self.config.depth - 1:
            #     _num_sums_out = self.config.num_classes
            # else:
            #     _num_sums_out = self.config.num_sums
            _num_sums_out = self.config.num_sums

            if self.config.layer_type == "einsum":
                layer = EinsumLayer(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            elif self.config.layer_type == "linsum":
                layer = LinsumLayer(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            elif self.config.layer_type == "linsum2":
                layer = LinsumLayer2(
                    num_features=in_features,
                    num_sums_in=_num_sums_in,
                    num_sums_out=_num_sums_out,
                    num_repetitions=self.config.num_repetitions,
                    dropout=self.config.dropout,
                )
            else:
                raise ValueError(f"Unknown layer type {self.config.layer_type}")

            # Update number of input features: each layer merges two partitions
            in_features = layer.num_features_out

            intermediate_layers.append(layer)

        if self.config.depth == 0:
            # Create a single sum layer
            layer = SumLayer(
                num_sums_in=self.config.num_leaves,
                num_features=1,
                num_sums_out=self.config.num_classes,
                num_repetitions=self.config.num_repetitions,
                dropout=self.config.dropout,
            )
            intermediate_layers.append(layer)

        # Construct final root product layer
        root_sum = SumLayer(
            num_sums_in=_num_sums_out,
            num_sums_out=self.config.num_classes,
            num_features=intermediate_layers[-1].num_features_out,
            num_repetitions=self.config.num_repetitions,
        )
        root_product = RootProductLayer(
            num_features=intermediate_layers[-1].num_features_out, num_repetitions=self.config.num_repetitions
        )

        intermediate_layers.append(root_sum)
        intermediate_layers.append(root_product)

        # Construct leaf
        leaf_num_features_out = self.config.num_features
        self.leaf = self._build_input_distribution_bottom_up()
        # self.leaf = self._build_input_distribution(num_features_out=leaf_num_features_out)

        # List layers in a bottom-to-top fashion
        self.layers: List[Union[EinsumLayer, LinsumLayer]] = nn.ModuleList(intermediate_layers)

        # Construct num_repertitions number of random permuations
        permutations = torch.empty((self.config.num_repetitions, self.config.num_features), dtype=torch.long)
        permutations_inv = torch.empty_like(permutations)
        for i in range(self.config.num_repetitions):
            permutations[i] = torch.randperm(self.config.num_features)
            permutations_inv[i] = invert_permutation(permutations[i])

        # Construct inverse permutations

        self.register_buffer("permutation", permutations)
        self.register_buffer("permutation_inv", permutations_inv)

        # If model has multiple reptitions, add repetition mixing layer
        if self.config.num_repetitions > 1:
            self.mixing = MixingLayer(
                num_features=1,
                num_sums_in=self.config.num_repetitions,
                num_sums_out=self.config.num_classes,
                dropout=self.config.dropout,
            )

        # Construct sampling root with weights according to priors for sampling
        if self.config.num_classes > 1:
            self._class_sampling_root = SumLayer(
                num_sums_in=self.config.num_classes,
                num_features=1,
                num_sums_out=1,
                num_repetitions=1,
            )
            self._class_sampling_root.weights = nn.Parameter(
                torch.log(
                    torch.ones(size=(1, self.config.num_classes, 1, 1)) * torch.tensor(1 / self.config.num_classes)
                ),
                requires_grad=False,
            )

    def _build_input_distribution_bottom_up(self) -> AbstractLeaf:
        """Construct the input distribution layer. This constructs a direct leaf and not a FactorizedLeaf since the bottom-up approach does not factorize."""
        # Cardinality is the size of the region in the last partitions
        return self.config.leaf_type(
            num_features=self.config.num_features,
            num_channels=self.config.num_channels,
            num_leaves=self.config.num_leaves,
            num_repetitions=self.config.num_repetitions,
            **self.config.leaf_kwargs,
        )

    def _build_input_distribution(self, num_features_out: int) -> FactorizedLeafSimple:
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        base_leaf = self.config.leaf_type(
            num_features=self.config.num_features,
            num_channels=self.config.num_channels,
            num_leaves=self.config.num_leaves,
            num_repetitions=self.config.num_repetitions,
            **self.config.leaf_kwargs,
        )

        if self.config.num_repetitions == 1:
            factorized_leaf_class = FactorizedLeafSimple
        else:
            factorized_leaf_class = FactorizedLeaf

        # factorized_leaf_class = FactorizedLeaf
        return factorized_leaf_class(
            num_features=base_leaf.out_features,
            num_features_out=num_features_out,
            num_repetitions=self.config.num_repetitions,
            base_leaf=base_leaf,
        )

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return next(self.parameters()).device

    def mpe(
        self,
        evidence: torch.Tensor = None,
        marginalized_scopes: List[int] = None,
        is_differentiable: bool = False,
    ) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(
            evidence=evidence, is_mpe=True, marginalized_scopes=marginalized_scopes, is_differentiable=is_differentiable
        )

    def sample(
        self,
        num_samples: Optional[int] = None,
        class_index=None,
        evidence: Optional[torch.Tensor] = None,
        is_mpe: bool = False,
        mpe_at_leaves: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: Optional[List[int]] = None,
        is_differentiable: bool = False,
        return_leaf_params: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `num_samples`: Generates `num_samples` samples.
        - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            num_samples: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            mpe_at_leaves: Flag to perform mpe only at leaves.
            marginalized_scopes: List of scopes to marginalize.
            is_differentiable: Flag to enable differentiable sampling.
            return_leaf_params: Flag to return the leaf distribution instead of the samples.
            seed: Seed for torch.random.

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        assert num_samples is None or evidence is None, "Cannot provide both, number of samples to generate (num_samples) and evidence."
        if self.config.num_classes == 1:
            assert class_index is None, "Cannot sample classes for single-class models (i.e. num_classes must be 1)."

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        if is_differentiable:
            indices_out = torch.ones(
                size=(num_samples, 1, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
            indices_repetition = torch.ones(
                size=(num_samples, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
        else:
            indices_out = torch.zeros(size=(num_samples, 1), dtype=torch.long, device=self.__device)
            indices_repetition = torch.zeros(size=(num_samples,), dtype=torch.long, device=self.__device)

        ctx = SamplingContext(
            num_samples=num_samples,
            is_mpe=is_mpe,
            mpe_at_leaves=mpe_at_leaves,
            temperature_leaves=temperature_leaves,
            temperature_sums=temperature_sums,
            num_repetitions=self.config.num_repetitions,
            evidence=evidence,
            indices_out=indices_out,
            indices_repetition=indices_repetition,
            is_differentiable=is_differentiable,
            return_leaf_params=return_leaf_params,
        )
        with sampling_context(self, evidence, marginalized_scopes, requires_grad=is_differentiable, seed=seed):
            if self.config.num_classes > 1:
                # If class is given, use it as base index
                if class_index is not None:
                    # Construct indices tensor based on given classes
                    if isinstance(class_index, list):
                        # A list of classes was given, one element for each sample
                        indices = torch.tensor(class_index, device=self.__device).view(-1, 1)
                        if is_differentiable:
                            # TODO: Test this
                            # One hot encode
                            indices = torch.zeros(
                                size=(num_samples, self.config.num_classes, 1), dtype=torch.float, device=self.__device
                            ).scatter_(1, indices.unsqueeze(-1), 1)
                            indices.requireds_grad_(True)  # Enable gradients
                        num_samples = indices.shape[0]
                    else:
                        indices = torch.empty(size=(num_samples, 1), dtype=torch.long, device=self.__device)
                        indices.fill_(class_index)
                        if is_differentiable:
                            # TODO: Test this
                            # One hot encode
                            indices = torch.zeros(
                                size=(num_samples, self.config.num_classes, 1), dtype=torch.float, device=self.__device
                            ).scatter_(1, indices.unsqueeze(-1), 1)
                            indices.requires_grad_(True)  # Enable gradients

                    ctx.indices_out = indices
                else:
                    # Sample class index from root
                    ctx = self._class_sampling_root.sample(ctx=ctx)

            # Save parent indices that were sampled from the sampling root
            if self.config.num_repetitions > 1:
                indices_out_pre_root = ctx.indices_out
                ctx = self.mixing.sample(ctx=ctx)

                # Obtain repetition indices
                if is_differentiable:
                    ctx.indices_repetition = ctx.indices_out.view(num_samples, self.config.num_repetitions)
                else:
                    ctx.indices_repetition = ctx.indices_out.view(num_samples)
                ctx.indices_out = indices_out_pre_root

            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self.layers):
                ctx = layer.sample(ctx=ctx)

            # Apply inverse permutation
            if hasattr(self, "permutation_inv"):
                # Select relevant inverse permuation based on repetition index
                if is_differentiable:
                    permutation_inv = self.permutation_inv.unsqueeze(0)  # Make space for num_samples
                    permutation_inv = self.permutation_inv.expand(num_samples, -1, -1)  # [N, R, D]
                    r_idxs = ctx.indices_repetition.unsqueeze(-1)  # Make space for feature dim
                    permutation_inv = index_one_hot(permutation_inv, r_idxs, dim=1)  # [N, D]
                    permutation_inv = permutation_inv.unsqueeze(-1).expand(-1, -1, self.config.num_leaves).long()  # [N, D, I]
                    ctx.indices_out = ctx.indices_out.gather(index=permutation_inv, dim=1)
                else:
                    permutation_inv = self.permutation_inv[ctx.indices_repetition]
                    ctx.indices_out = ctx.indices_out.gather(index=permutation_inv, dim=1)

            # Sample leaf
            samples = self.leaf.sample(ctx=ctx, cont_shift_params=self.cont_shift_parameters, disc_shift_params=self.disc_shift_parameters)

            if return_leaf_params:
                # Samples contain the distribution parameters instead of the samples
                return samples

            if evidence is not None:
                # First make a copy such that the original object is not changed
                evidence = evidence.clone().float()
                shape_evidence = evidence.shape
                evidence = evidence.view_as(samples)

                if marginalized_scopes is None:
                    mask = torch.isnan(evidence)
                    evidence[mask] = samples[mask].to(evidence.dtype)
                else:
                    evidence[:, :, marginalized_scopes] = samples[:, :, marginalized_scopes].to(evidence.dtype)

                evidence = evidence.view(shape_evidence)
                return evidence
            else:
                return samples

    def extra_repr(self) -> str:
        return f"{self.config}"


def posterior(ll_x_g_y: torch.Tensor, num_classes) -> torch.Tensor:
    """
    Compute the posterior probability logp(y | x) of the data.

    Args:
        x: Data input.

    Returns:
        Posterior logp(y | x).
    """
    # logp(y | x) = logp(x, y) - logp(x)
    #             = logp(x | y) + logp(y) - logp(x)
    #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)
    ll_y = np.log(1.0 / num_classes)
    ll_x_and_y = ll_x_g_y + ll_y
    ll_x = torch.logsumexp(ll_x_and_y, dim=1, keepdim=True)
    ll_y_g_x = ll_x_g_y + ll_y - ll_x
    return ll_y_g_x
