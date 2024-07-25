from typing import Dict, List
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
import numpy as np
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context, Product, Sum, Leaf
from spn.structure.leaves.parametric.Parametric import Categorical, CategoricalDictionary
from spn.structure.StatisticalTypes import MetaType
from spn.structure.Base import get_nodes_by_type
from spn.algorithms.Marginalization import marginalize
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
import scipy
import warnings

warnings.filterwarnings("ignore")

class PCOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100, 
                 samples_per_iter=20, initial_samples=20, use_eic=False, eic_samplings=20,
                 deactivate_transfer=False) -> None:
        super().__init__(search_space, objective)
        self.curr_pc = None
        self.modified_pc = None # for interactive part and adaptation
        self._num_iterations = iterations
        self._samples_per_iter = samples_per_iter
        self._initial_samples = initial_samples
        self._rand_state = RandomState(123)
        self._use_eic = use_eic
        self._eic_samplings = eic_samplings
        self._deactivate_transfer = deactivate_transfer # if true, will act as PC optimizer
        # must be recorded for logging
        self._curr_iter = 0
        self._construct_search_space_metadata()
        self._old_search_space = None
        self._search_space_switched = False
        self._global_pc = None
        self._seen_dimensions = search_space.get_search_space_definition()
        self._first_and_last_sample = []

    def optimize(self):
        """
            Fit a PC and use it to sample new promising configurations
        """
        for i in range(self._num_iterations):
            self._curr_iter = i
            if i % 10 == 0:
                print(f"Iteration {i+1}/{self._num_iterations}")

            if self.curr_pc is None:
                self._learn_init_pc()
            elif self._search_space_switched:
                self._learn_trans_init_pc()
            else:
                self._learn_pc()
            self._sample()
        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        if self._global_pc is not None and not self._deactivate_transfer:
            self._fuse_distributions()
        else:
            self._global_pc = self.curr_pc
        return self.evaluations[max_idx][:2]

    def set_search_space(self, new_search_space: SearchSpace):
        """
            Set new search space and find out which variables have beend removed and which have been added.
        """
        # if transfer is deactivated, reset PC and act as normal PC optimizer
        if self._deactivate_transfer:
            self.evaluations = []
            self.curr_pc = None
            self.search_space = new_search_space
            self._construct_search_space_metadata()
            return
        new_search_space_dimensions = set(new_search_space.get_search_space_definition().keys())
        
        # find new variables and unused variables
        seen_vars = set(self._seen_dimensions.keys())
        self.added_vars = list(new_search_space_dimensions.difference(seen_vars))
        self.removed_vars = list(seen_vars.difference(new_search_space_dimensions))

        # adapt global PC to new search sapce and set as current PC
        marg_pc = self._marginalize()
        self.curr_pc = self._cmopute_product_distribution(marg_pc, new_search_space)

        # set new context for PC
        self._seen_dimensions = {**self._seen_dimensions, **new_search_space.get_search_space_definition()}
        self._construct_search_space_metadata()
        self._search_space_switched = True
        self.evaluations = []

    def _fuse_distributions(self):
        # TODO: how to transfer knowledge, do we have to drop correlation knowledge?
        # 1. compute product distribution of current global PC and newly added dimensions.
        #   TODO: Think about doing product with marginal rather than uniform for new dimensions?
        prod = self._cmopute_product_distribution(self._global_pc, self.search_space)
        # copy the PC as we do one marginalized version of it
        
        # 2. approximate second mixture component using rehearsal
        # sample from current pc
        if len(self.removed_vars) > 0:
            vars = list(self.search_space.get_search_space_definition().keys())
            cond = np.repeat([np.nan], 5000*len(vars)).reshape(5000, len(vars))
            curr_samples = sample_instances(self.curr_pc, cond, self._rand_state)
            seen_vars = list(self._seen_dimensions.keys())
            cond_idx = [seen_vars.index(i) for i in vars]
            samples = []
            for sample in curr_samples:
                cond_arr = np.array([[np.nan] * len(seen_vars)])
                for cidx, s in zip(cond_idx, sample):
                    cond_arr[cidx] = s
                final_sample = sample_instances(self._global_pc, cond_arr, self._rand_state)[0]
                samples.append(final_sample)
            samples = np.array(samples)
            meta_types = [val['type'] for val in self._seen_dimensions.values()]
            meta_types += [MetaType.REAL]
            ctxt = Context(meta_types=meta_types)
            scd_pc = learn_mspn(samples, ctxt, min_instances_slice=80)
        else:
            scd_pc = self.curr_pc

        # 3. merge distributions into mixture
        fs, ls = self._first_and_last_sample
        fs_r, ls_r = fs[:, -1], ls[:, -1]
        avg_optim_gain = np.mean(abs(fs_r - ls_r))
        w1 = sigmoid(avg_optim_gain)
        w2 = 1 - w1
        root = Sum([w1, w2], [prod, scd_pc])
        self._global_pc = root

    def _marginalize(self):
        if len(self.removed_vars) == 0:
            return self._global_pc
        seen_dims = list(self._seen_dimensions.keys())
        marg_idx = [seen_dims.index(a) for a in self.added_vars]
        marg_pc = marginalize(self._global_pc, marg_idx)
        return marg_pc
    
    def _cmopute_product_distribution(self, pc, new_search_space: SearchSpace):
        if len(self.added_vars) == 0:
            return pc
        # get set of existing scope identifiers
        nodes = get_nodes_by_type(pc)
        scopes = list(set([node.scope for node in nodes]))
        max_scope = max(scopes)

        # add product distribution
        ssd = new_search_space.get_search_space_definition()
        prod_children = []
        for i, d in enumerate(self.added_vars):
            scope = max_scope + (i + 1)
            scopes.append(scope)
            if ssd[d]['type'] == MetaType.DISCRETE:
                vals = ssd[d]['allowed']
                p = [1/len(vals)] * len(vals)
                node = Categorical(p=p, scope=scope)
            elif ssd[d]['type'] == MetaType.REAL:
                # since Uniform distribution from SPFlow is not documented, we approximate it by samples
                min_, max_ = ssd[d]['interval']
                choices = np.random.uniform(min_, max_, size=1000)
                probs = [1/1000] * 1000
                prob_dict = {v: p for p, v in zip(probs, choices)}
                node = CategoricalDictionary(p=prob_dict, scope=scope)
            prod_children.append(node)

        prod_node = Product([pc] + prod_children)
        prod_node.scope = scopes
        return prod_node

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

    def _learn_trans_init_pc(self, k=5):
        assert int(round(self._samples_per_iter / k)) == int(self._samples_per_iter / k), 'k must divide samples per iteration'
        # use current PC to sample some data that is transferred to the next search space
        # TODO: marginalize out score dimension
        # 1. sample from adapted PC (done by set_search_sapce) where score is marginalized out
        init_samples = k*self._samples_per_iter
        pc_dims = len(list(self.search_space.get_search_space_definition().keys())) + 1
        cond_array = np.repeat(np.nan, pc_dims*init_samples).reshape(init_samples, pc_dims)
        samples = sample_instances(self.curr_pc, cond_array, self._rand_state)

        # only keep k=5 best and sample conditioned on these
        # TODO: for this work reliably, we have to ensure that all scores are in the same range!
        #   Or condition on some task descriptor
        keep_idx = np.argsort(samples[:,-1])[-k:]
        cond_scores = samples[:,-1][keep_idx]
        cond_array = np.repeat(np.nan, self._samples_per_iter*pc_dims).reshape(self._samples_per_iter, pc_dims)
        cond_array[:,-1] = np.repeat(cond_scores, int(self._samples_per_iter / k))
        samples = sample_instances(self.curr_pc, cond_array, self._rand_state)[:,:-1]

        # 2. evaluate sampled configurations
        configs = []
        for sample in samples:
            d = {n: v for n, v in zip(self.hyperparam_names, sample)}
            configs.append(d)
        scores = [self.objective(cfg) for cfg in configs]
        val_scores = np.array([s.val_performance for s in scores])
        self.data = np.column_stack((samples, val_scores.reshape(-1, 1)))

        # 4. fit PC
        self._learn_pc()
        self._search_space_switched = False # reset switch of search space indicator

    def _learn_pc(self):
        # drop duplicates
        data = np.unique(self.data, axis=0)
        self.ctxt.add_domains(data)
        mu_hp = max(20, data.shape[0] // 100)
        self.curr_pc = learn_mspn(data, self.ctxt, min_instances_slice=mu_hp)

    def _sample(self):
        cond_array = [np.nan] * (self.data.shape[1])
        cond_array[-1] = self.data[:, -1].max()
        # conditional sampling
        samples = sample_instances(self.curr_pc, np.array([cond_array] * self._samples_per_iter), self._rand_state)
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
        if len(self._first_and_last_sample) in [0, 1]:
            self._first_and_last_sample.append(evaled_samples)
        else:
            self._first_and_last_sample[-1] = evaled_samples

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