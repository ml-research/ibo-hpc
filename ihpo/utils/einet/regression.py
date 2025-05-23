from .einet import Einet
from ihpo.search_spaces import SearchSpace
from ihpo.utils import ConfigurationNumericalTransform
from ihpo.utils.einet.dist import Domain
from typing import List, Tuple
import torch

class EvolutionaryOptimizationEinetRegressionEngine:

    def __init__(self, model_history: List[Tuple[Einet, Domain, SearchSpace, ConfigurationNumericalTransform]], curr_search_sapce: SearchSpace,
                 mixture_weights: torch.FloatTensor, cont_shift_params: torch.FloatTensor, disc_shift_params: List[torch.FloatTensor], 
                 function_idx: int):
        
        self.curr_search_space = curr_search_sapce
        self.model_history = model_history
        self.mixture_weights = mixture_weights
        self.cont_shift_params = cont_shift_params
        self.disc_shift_params = disc_shift_params
        self.function_idx = function_idx
        self._infer_marginalization_indices()

    def _infer_marginalization_indices(self):
        """For each task search space, compute the marginalization indices w.r.t. the current search space.
        I.e., compute which variables have to be marginalized out for each task when doing predictions.
        """

        self.marginalization_indices = []
        curr_keys = set(list(self.curr_search_space.get_search_space_definition().keys()))

        for _, _, task_search_space, _, _ in self.model_history:
            hp_intersection = SearchSpace.get_intersection_keys(task_search_space, self.curr_search_space)
            curr_keys = list(curr_keys)
            marg_key_idx = [idx for idx, _ in enumerate(curr_keys) if curr_keys[idx] not in hp_intersection]
            if len(marg_key_idx) == 0:
                self.marginalization_indices.append(None)
            else:
                self.marginalization_indices.append(marg_key_idx)

    def predict(self, hp_vector: list, mpe=True):
        """
            Predict MPE given a vector of HPs.
            We cannot put this vector into the einet directly as it contains values directly sampled from the
            search space definition. However, we apply an encoding before feeding it into the einet.
            To keep the code of the evolutionary optimization and the fitness function clean, we do the pre-processing and sampling here.

        Args:
            hp_vector (list): List representation of a hyperparameter configuration.
        """

        model_preds = []
        for idx, (einet, _, task_search_space, transform, _) in enumerate(self.model_history):

            cfg_dict = {k: v for k, v in zip(task_search_space.get_search_space_definition().keys(), hp_vector)}
            processed_hp_vector = transform.transform_configuration(cfg_dict) + [0] # transform cfg and append placeholder for score

            cond_array = torch.tensor(processed_hp_vector).to(torch.float32).reshape(1, -1)
            cond_array[:, -1] = torch.nan
            # setting the variables to be marginalized to nan
            # this works because the conditioning and marginalization mechanisms in einet are the same
            cond_array[:, self.marginalization_indices[idx]] = torch.nan

            cont_shift_params = self.cont_shift_params[idx]
            disc_shift_params = self.disc_shift_params[idx]

            einet.init_shift_parameters()
            if len(cont_shift_params) > 0:
                einet.cont_shift_parameters.data = cont_shift_params[self.function_idx].unsqueeze(0)

            if len(disc_shift_params) > 0:
                for einet_disc_param, disc_param in zip(einet.disc_shift_parameters, disc_shift_params[self.function_idx]):
                    einet_disc_param.data = disc_param.unsqueeze(0)
            pred = einet.sample(evidence=cond_array, is_mpe=mpe).squeeze()[-1]
            model_preds.append(pred)
        
        model_preds = torch.tensor(model_preds).to(torch.float32)
        w = torch.softmax(self.mixture_weights[self.function_idx], dim=-1)
        result = torch.sum(w*model_preds)
        if len(result.shape) == 0:
             result = torch.tensor([result])
        return result
    

class MixtureEinetRegressionEngine:

    def __init__(self, model_history: List[Tuple[Einet, Domain, SearchSpace, ConfigurationNumericalTransform]], curr_search_sapce: SearchSpace,
                 mixture_weights: torch.FloatTensor):
        
        self.curr_search_space = curr_search_sapce
        self.model_history = model_history
        self.mixture_weights = mixture_weights
        self._infer_marginalization_indices()

    def _infer_marginalization_indices(self):
        """For each task search space, compute the marginalization indices w.r.t. the current search space.
        I.e., compute which variables have to be marginalized out for each task when doing predictions.
        """

        self.set_indices = []
        curr_keys = set(list(self.curr_search_space.get_search_space_definition().keys()))

        for _, _, task_search_space, _, _ in self.model_history:
            hp_intersection = SearchSpace.get_intersection_keys(task_search_space, self.curr_search_space)
            curr_keys = list(curr_keys)
            marg_key_idx = [idx for idx, _ in enumerate(curr_keys) if curr_keys[idx] in hp_intersection]
            if len(marg_key_idx) == 0:
                self.set_indices.append(None)
            else:
                self.set_indices.append(marg_key_idx)
    
    def predict(self, configs: torch.FloatTensor, mpe=True):
        """
            Tensoirzed version of predict.
            
            NOTE: Assumes the configurations in `configs` are already transformed.

        Args:
            configs (torch.FloatTensor): Configurations to be predicted
        """
        model_preds = []
        for idx, (einet, domains, _, _, _) in enumerate(self.model_history):

            # if task has a RV that is not in current search space, leave it as nan -> gets marginalized
            # if current search space has a RV that is not in task, ignore (equals adding uniform and marginalize uniform out again)
            data = torch.full((configs.shape[0], len(domains)), float('nan')).to(torch.float32)
            data[:, :-1] = configs.squeeze()[:, self.set_indices[idx]].to(torch.float32)

            pred = einet.sample(evidence=data, is_mpe=mpe).squeeze()
            model_preds.append(pred)
        
        model_preds = torch.cat(model_preds, dim=1).to(torch.float32)
        w = torch.softmax(self.mixture_weights, dim=-1)
        return torch.sum(w*model_preds, dim=1).squeeze()