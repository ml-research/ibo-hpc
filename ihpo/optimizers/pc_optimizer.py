from typing import Dict, List
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
import numpy as np
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from numpy.random.mtrand import RandomState
import scipy
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")

class PCOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100, 
                 samples_per_iter=20, initial_samples=20, use_eic=False, eic_samplings=20,
                 interaction_dist_sample_decay=0.9, conditioning_value_quantile=1) -> None:
        super().__init__(search_space, objective)
        self.curr_pc = None
        self.modified_pc = None # for interactive part and adaptation
        self.search_space = search_space
        self.objective = objective
        self._num_iterations = iterations
        self._samples_per_iter = samples_per_iter
        self._initial_samples = initial_samples
        self._rand_state = RandomState(123)
        self._use_eic = use_eic
        self._eic_samplings = eic_samplings
        self.evaluations = []
        # must be recorded for logging
        self._curr_iter = 0
        self._construct_search_space_metadata()
        self._interventions = None
        self._intervention_iters: List[int] or None = None
        self._intervention_duration = None
        self._intervention = None
        self._prob_interaction_pc = 1
        self._interaction_dist_sample_decay = interaction_dist_sample_decay
        self._original_leaf_params: dict or None = None
        self._conditioning_value_quantile = conditioning_value_quantile
        if conditioning_value_quantile != 1:
            print("WARNING: PC Optimizer is in ablation mode. For best performance set conditioning_value_quantile=1.")

    def optimize(self):
        """
            Fit a PC and use it to sample new promising configurations
        """
        intervene = False
        for i in range(self._num_iterations):
            self._curr_iter = i
            if i % 10 == 0:
                print(f"Iteration {i+1}/{self._num_iterations}")
            
            if self._intervention_iters is not None and self._interventions is not None:# and self._intervention_duration is not None:
                if i in self._intervention_iters:
                    int_idx = self._intervention_iters.index(i)
                    print(f"[Optimizer][PC] Using Intervention {int_idx}: {self._interventions[int_idx]}")
                    self._intervention = self._interventions[int_idx]
                    intervene = True and self._intervention is not None

            if self.curr_pc is None:
                self._learn_init_pc()
                self._sample(intervene=intervene)
            else:
                self._learn_pc()
                self._sample(intervene=intervene)
        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        return self.evaluations[max_idx][:2]


    def intervene(self, interventions: List[Dict], iters):
        """
            Set certain dimensions to a fixed value to obtain a conditional distribution
        """
        self._interventions = interventions
        self._intervention_iters = iters

    def adapt_to_search_space(self, new_search_space: Dict):
        """
            Adapt PC to a new search space by marginalizing unused variables out.

        """
        raise NotImplementedError()

    def _apply_point_intervention(self, data):
        # get prototype configuration to determine scope id of hyperparameter in PC.
        prot_cfg = self.search_space.sample(keep_domain=False, size=1)[0]
        keys = list(self._intervention.keys())
        if self._intervention is not None:
            for idx, key in enumerate(prot_cfg.keys()):
                if key in keys:
                    data[idx] = self._intervention[key]

        return data

    def _apply_intervention(self):

        def sample_from_condition_prior(intervention_key):
            if self._intervention[intervention_key]['dist'] == 'gauss':
                mu, std = self._intervention[intervention_key]['parameters']
                samples = np.random.normal(mu, std, self._samples_per_iter)
            elif self._intervention[intervention_key]['dist'] == 'uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = np.random.uniform(s, e, self._samples_per_iter)
            elif self._intervention[intervention_key]['dist'] == 'cat':
                weights = self._intervention[intervention_key]['parameters']
                probs = np.array(weights) / np.sum(weights)
                samples = np.random.choice(np.arange(len(probs)), self._samples_per_iter, p=probs)
            return samples

        # get prototype configuration to determine scope id of hyperparameter in PC.
        prot_cfg = self.search_space.sample(keep_domain=False, size=1)[0]
        keys = list(self._intervention.keys())
        cond_samples = np.array([np.nan]*self._samples_per_iter*(len(prot_cfg) +1)).reshape(-1, len(prot_cfg) +1)
        if self._intervention is not None:
            for idx, key in enumerate(prot_cfg.keys()):
                if key in keys:
                    if isinstance(self._intervention[key], dict):
                        # dict means we have a distribution specified
                        cond_samples[:, idx] = sample_from_condition_prior(key)
                    else:
                        # else we have single point specified
                        cond_samples[:, idx] = np.repeat(self._intervention[key], self._samples_per_iter)

        return cond_samples

    def _build_dataset(self, configs, scores):
        """
            build a data matrix based on configuration and score pairs
        """
        data_matrix = []
        for cfg, score in zip(configs, scores):
            features = [val for val in cfg.values()]
            features.append(score)
            data_matrix.append(features)
        return np.array(data_matrix)
    
    def _learn_init_pc(self):
        # random sampling
        configs = self.search_space.sample(keep_domain=False, size=self._initial_samples)
        scores = [self.objective(cfg) for cfg in configs]
        val_scores = [s.val_performance for s in scores]
        self.data = self._build_dataset(configs, val_scores)
        self._learn_pc()

    def _learn_pc(self):
        # drop duplicates
        data = np.unique(self.data, axis=0)
        self.ctxt.add_domains(data)
        mu_hp = max(80, data.shape[0] // 100)
        # NOTE: If this fails with memory error, try rounding the performances
        self.curr_pc = learn_mspn(data, self.ctxt, min_instances_slice=mu_hp)
    
    def _get_conditioning_value(self):
        """
            Only used for ablation studies to analyze importance of conditioning on the best value.
            If optimizer is running in normal mode, set self._conditioning_value_quantile=1.
        """
        if self._conditioning_value_quantile == 1:
            return self.data[:, -1].max()
        else:
            return np.quantile(self.data[:, -1], self._conditioning_value_quantile)

    def _sample(self, intervene=False):
        cond_array = np.array([np.nan] * self._samples_per_iter * self.data.shape[1]).reshape(-1, self.data.shape[1])
        cond_array[:, -1] = self._get_conditioning_value()
        if intervene:
            accept_intervention = np.random.choice([1, 0], p=[self._prob_interaction_pc, 1-self._prob_interaction_pc])
            self._prob_interaction_pc *= self._interaction_dist_sample_decay
            if accept_intervention == 1:
                cond_array = self._apply_intervention()
            cond_array[:, -1] = self._get_conditioning_value()
        # conditional sampling
        samples = sample_instances(self.curr_pc, np.array(cond_array), self._rand_state)
        if self._use_eic:
            samples = self._filter_eic(samples)
        
        # evaluate samples
        sampled_configs = samples[:, :-1]
        evaled_samples = []
        for i in range(len(sampled_configs)):
            sample = list(sampled_configs[i])
            # if already sampled, skip
            #if np.any(np.all(self.data[:, :-1] == np.array(sample), axis=1)):
            #    continue
            config_dict = {n: s for n, s in zip(self.hyperparam_names, sample)}
            evaluation = self.objective(config_dict)
            if evaluation is not None:
                self.evaluations.append((config_dict, evaluation, self._curr_iter))
                sample.append(evaluation.val_performance)
                evaled_samples.append(sample)
        evaled_samples = np.array(evaled_samples)
        if evaled_samples.shape[0] > 0:
            # may be that we have seen all sampled configs already
            self.data = np.concatenate((self.data, evaled_samples))

    def _filter_eic(self, samples):
        preds_matrix = np.empty((self._samples_per_iter, self._eic_samplings))
        cond_matrix = samples.copy()
        cond_matrix[:, -1] = np.nan

        for i in range(self._eic_samplings):
            preds_matrix[:, i] = sample_instances(self.curr_pc, cond_matrix, self._rand_state)[:, -1]
        error_matrix = 100 - preds_matrix.copy()
        best_so_far = 100 - self.data[:, -1].max()
        mean_current = error_matrix.mean(axis=1)
        std_current = error_matrix.std(axis=1)

        var_current = std_current ** 2
        u = (best_so_far - mean_current) / std_current
        ei = var_current * [(u ** 2 + 1) * scipy.stats.norm.cdf(u) + u * scipy.stats.norm.pdf(u)]
        # rank by EI and return top k configurations
        rank_ind = np.argsort(ei.flatten(), axis=0)
        samples = np.take_along_axis(samples, rank_ind.reshape(-1,1), axis=0)
        n_picked_samples = samples.shape[0] // 2  # TODO take 50%
        samples = samples[n_picked_samples:]
        return samples
    
    def _construct_search_space_metadata(self):
        search_space_definition = self.search_space.get_search_space_definition()
        meta_types = [val['type'] for val in search_space_definition.values()]
        meta_types += [MetaType.REAL]
        self.hyperparam_names = list(search_space_definition.keys())
        self.ctxt = Context(meta_types=meta_types)

    @property
    def history(self):
        return self.evaluations