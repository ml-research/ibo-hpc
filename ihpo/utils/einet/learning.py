from .einet import Einet, EinetConfig
from .layers.distributions.piecewise_linear import PiecewiseLinear
from .dist import Domain, DataType
from ..losses import StatefulDifferentiableRankingLoss
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from typing import List
from ihpo.search_spaces import SearchSpace
import torch

def train_einet(data: torch.FloatTensor, domains: List[Domain], epochs=100, batch_size=128, lr=0.1, 
                max_num_sums=5, max_num_leaves=2, max_num_repitions=5, laplace_smoothing_alpha=0.0, log_loss=False, gpu=None):
    """Learn the parameter of an einsum network s.t. it represents the joint distribution `data` was sampled from as good as possible.
    To this end, we maximize the log likelihood (minimize neg. LL).

    Args:
        data (TensorDataset): Data collected by evaluating the objective function.
        domains (List, Domain): Domain of the search space.
        epochs (int, optional): Number of epochs. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 128.
        lr (float, optional): Learning rate. Defaults to 0.1.
        max_num_sums (int, optional): Number of sum nodes per layer.
        max_num_leaves (int, optional): Number of leaf nodes.
        max_num_repititions (int, optional): Number of repitiions of the einsum network.
        log_loss (bool, optional): Log loss? Defaults to False.
        gpu (int, optional): Use GPU. Defaults to None.

    Returns:
        Einet: Learned einet.
    """
    einet_hps = Einet.get_hyperparameters(len(domains), len(data))
    cfg = EinetConfig(
            num_features=len(domains),
            depth=einet_hps['depth'],
            num_sums=min(einet_hps['num_sums'], max_num_sums),
            num_channels=1,
            num_leaves=min(einet_hps['num_leaves'], max_num_leaves),
            num_repetitions=min(einet_hps['num_repitions'], max_num_repitions),
            num_classes=1,
            dropout=0.0,
            leaf_type=PiecewiseLinear,
            layer_type='einsum',
            leaf_kwargs={'alpha': laplace_smoothing_alpha}
        )
    einet = Einet(cfg, domains)
    einet.leaf.base_leaf.initialize(data, domains=domains)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size)
    optim = Adam(einet.parameters(), lr=lr)

    for epoch in range(epochs):
        total_ll = 0.0
        for x in loader:
            x = x[0]
            optim.zero_grad()

            # Forward pass: compute log-likelihoods
            lls = einet(x)

            # Backprop negative log-likelihood loss
            nlls = -1 * lls.sum()
            nlls.backward()
            total_ll += lls.sum().item()
            # Update weights
            optim.step()
        
        if log_loss:
            print(f"[Learn Einet] LL: {total_ll}")
    
    return einet


def train_shift_and_mixture_model(einets: List[Einet], search_spaces: List[SearchSpace], curr_search_space: SearchSpace, data: torch.FloatTensor, 
                                  mixture_weights: torch.nn.Parameter, epochs=100, lr=0.1, log_loss=False, gpu=None):
    """
    Optimize mixture weights and shift parameters of Einets using a differentiable ranking loss.
    This optimization enables knowledge transfer by shifting and weighting the distributions represented by the Einets
    s.t. the conditional p(y|x) reflects the datapoints from the current task well. Here, y is the evaluation score and x are hyperparameters.  

    Args:
        einets (List[Einet]): List of einsum networks representing joint distributions over former search spaces.
        search_spaces (List[SearchSpace]): List of search spaces the einets were trained on.
        curr_search_space: SearchSpace: Search space of the current task.
        data (torch.FloatTensor): Data collected on the new search space (usually a few true observations and some sampled; see paper for details).
        mixture_weights (torch.nn.Parameter): Weights of conditional p(y|x) to represent new data points as closely as possible according to ranking loss.
        epochs (int, optional): Number of epochs to train. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 0.1.
        log_loss (bool, optional): Should the loss be logged? Defaults to False.
        gpu (int, optional): GPU to use. Defaults to None.

    Returns:
        Tuple: Learned mixture weights and einets with optimized shift parameters.
    """

    points_copy = data.clone().squeeze()
    points_copy[:, -1] = torch.nan
    shift_params = []
    for einet in einets:
        einet.init_shift_parameters()
        shift_params.append(einet.cont_shift_parameters)
        shift_params += einet.disc_shift_parameters
    optimizer = torch.optim.Adam([mixture_weights] + shift_params, lr)
    schedule = torch.optim.lr_scheduler.LinearLR(optimizer, 1., 1/3, epochs)
    ranking_loss = StatefulDifferentiableRankingLoss()

    datasets = []
    for search_space in search_spaces:
        # prepare data
        intersection_hps = SearchSpace.get_intersection_keys(curr_search_space, search_space)

        einet_data = torch.full((points_copy.shape[0], len(search_space.get_search_space_definition().keys()) + 1), float('nan')).to(torch.float32)
        search_space_idx = [idx for idx, key in enumerate(search_space.get_search_space_definition().keys()) if key in intersection_hps]
        curr_space_idx = [idx for idx, key in enumerate(curr_search_space.get_search_space_definition().keys()) if key in intersection_hps]
        einet_data[:, search_space_idx] = points_copy[:, curr_space_idx].to(torch.float32)

        datasets.append(einet_data)

    for _ in range(epochs):
        optimizer.zero_grad()

        function_samples = torch.zeros(points_copy.shape[0], len(einets))
        for i, (einet, dataset) in enumerate(zip(einets, datasets)):

            # get function values predicted by conditioning the einet on hyperparameter values and sample
            function_samples[:,i] = einet.sample(evidence=dataset, is_differentiable=True, is_mpe=True).squeeze()[:,-1]

        w = torch.softmax(mixture_weights, dim=0)

        # compute weighted average of samples
        samples = torch.sum(function_samples * w, dim=1)
        loss = ranking_loss(data.squeeze()[:, -1].squeeze(), samples.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_([mixture_weights] + shift_params, 1.)
        optimizer.step()
        schedule.step()

        if log_loss:
            print(f"[Transfer Knowledge] RANKING-LOSS: {loss.item()}")
        
    return mixture_weights, einets

def train_second_order_mixture_model(mixture_model: List[Einet], einet: Einet, search_spaces: List[SearchSpace], curr_search_space: SearchSpace, 
                                     data: torch.FloatTensor, mixture_weights: torch.nn.Parameter, second_order_mixture_weights: torch.nn.Parameter, 
                                     epochs=100, lr=0.1, log_loss=False, gpu=None):
    """
    Optimize second order mixture weights between a einet and a given mixture of einets.  

    Args:
        mixture_model (List[Einet]): List of einsum networks representing joint distributions over former search spaces.
        einet (Einet): Einsum network representing distribution over current search space.
        search_spaces (List[SearchSpace]): List of search spaces the einets were trained on.
        curr_search_space: SearchSpace: Search space of the current task.
        data (torch.FloatTensor): Data collected on the new search space (usually a few true observations and some sampled; see paper for details).
        mixture_weights (torch.nn.Parameter): Weights of conditional mixture p(y|x) that represent current data in terms of history models.
        second_order_mixture_weights (torch.nn.Parameter): Weights that assign importance to the einet and the mixture_model.
        epochs (int, optional): Number of epochs to train. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 0.1.
        log_loss (bool, optional): Should the loss be logged? Defaults to False.
        gpu (int, optional): GPU to use. Defaults to None.

    Returns:
        Tuple: Learned second order mixture weights.
    """

    points_copy = data.clone().squeeze()
    points_copy[:, -1] = torch.nan
    optimizer = torch.optim.Adam([second_order_mixture_weights], lr)
    ranking_loss = StatefulDifferentiableRankingLoss()

    datasets = []
    for search_space in search_spaces:
        # prepare data
        intersection_hps = SearchSpace.get_intersection_keys(curr_search_space, search_space)

        einet_data = torch.full((points_copy.shape[0], len(search_space.get_search_space_definition().keys()) + 1), float('nan')).to(torch.float32)
        search_space_idx = [idx for idx, key in enumerate(search_space.get_search_space_definition().keys()) if key in intersection_hps]
        curr_space_idx = [idx for idx, key in enumerate(curr_search_space.get_search_space_definition().keys()) if key in intersection_hps]
        einet_data[:, search_space_idx] = points_copy[:, curr_space_idx].to(torch.float32)

        datasets.append(einet_data)

    for _ in range(epochs):
        optimizer.zero_grad()

        
        second_order_function_samples = torch.zeros(data.shape[0], 2)
        function_samples = torch.zeros(data.shape[0], len(mixture_model))
        for i, (model, dataset) in enumerate(zip(mixture_model, datasets)):
            # get function values predicted by conditioning the einet on hyperparameter values and sample
            function_samples[:,i] = model.sample(evidence=dataset, is_differentiable=True, is_mpe=True).squeeze()[:,-1]

        w = torch.softmax(mixture_weights, dim=0)

        # compute weighted average of samples
        samples = torch.sum(function_samples * w, dim=1)
        second_order_function_samples[:, 1] = samples.squeeze()
        einet_samples = einet.sample(evidence=points_copy, is_differentiable=True, is_mpe=True).squeeze()[:,-1]
        second_order_function_samples[:, 0] = einet_samples.squeeze()

        scd_order_w = torch.softmax(second_order_mixture_weights, dim=0)
        scd_order_samples = torch.sum(second_order_function_samples * scd_order_w, dim=1)

        loss = ranking_loss(data.squeeze()[:, -1].squeeze(), scd_order_samples.squeeze())
        loss.backward()
        optimizer.step()

        if log_loss:
            print(f"[Transfer Knowledge] RANKING-LOSS: {loss.item()}")
        
    return second_order_mixture_weights