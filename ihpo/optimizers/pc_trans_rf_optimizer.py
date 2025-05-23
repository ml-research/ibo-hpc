from typing import Dict, List
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..utils import ConfigurationNumericalTransform
import numpy as np
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context, Product, Sum, Leaf, assign_ids
from spn.structure.leaves.parametric.Parametric import Categorical, CategoricalDictionary
from spn.structure.StatisticalTypes import MetaType
from spn.structure.Base import get_nodes_by_type
from spn.algorithms.Marginalization import marginalize
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from skopt.learning import RandomForestRegressor
from skopt import forest_minimize
from skopt.utils import use_named_args
import warnings

warnings.filterwarnings("ignore")

class PCTransRFOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100, 
                 warm_start_samples=1e4, global_conditional_approx_samples=1e4, seed=0) -> None:
        super().__init__(search_space, objective)
        self.curr_pc = None
        self.modified_pc = None # for interactive part and adaptation
        self._iterations = iterations
        self._warm_start_samples = int(warm_start_samples)
        self._global_conditional_approx_samples = global_conditional_approx_samples
        self._rand_state = RandomState(seed)
        # must be recorded for logging
        self._curr_iter = 0
        self._construct_search_space_metadata()
        self._old_search_space = None
        self._search_space_switched = False
        self._global_pc = None
        self._seen_dimensions = search_space.get_search_space_definition()
        self._first_and_last_sample = []
        self.transform = ConfigurationNumericalTransform(self.search_space)

    def optimize(self):
        """
            Perform BO using random forest as surrogate and EI as acquisition. Warm-start RF if a PC from a previous task is available before optimization.
        """
        skopt_space = self.search_space.to_skopt()

        @use_named_args(skopt_space)
        def _objective(**params):
            if self._curr_iter % 10 == 0:
                print(f"Iteration {self._curr_iter}/{self._iterations}")
            config = self.transform.inv_transform_configuration(list(params.values()))
            res = self.objective(config)
            self.evaluations.append((config, res, self._curr_iter))
            self._curr_iter += 1
            return -res.val_performance

        if self.curr_pc is not None and self._search_space_switched:
            # warm-start model
            warm_start_samples = self._sample()
            X, y = warm_start_samples[:, :-1], warm_start_samples[:, -1].reshape(-1, 1)
            X = np.array(X)

            for _ in range(self._iterations):
                if len(self.evaluations) > 0:
                    cfg, res = self.evaluations[-1][:2]
                    new_X = np.array([list(self.transform.transform_configuration(cfg))])
                    new_y = np.array([-res.val_performance])
                    X = np.concatenate((X, new_X), axis=0)
                    y = np.concatenate((y, new_y.reshape(-1, 1)), axis=0)
                rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1)
                rf.fit(X, y)

                # BO with warm-started RF
                forest_minimize(_objective, skopt_space, n_calls=1, base_estimator=rf, n_initial_points=1)
        else:
            # BO without warm-start since this is the first task we see
            forest_minimize(_objective, skopt_space, n_calls=self._iterations, base_estimator='rf')

        # learn/update current PC    
        self.data = self._build_dataset()
        self._learn_pc()
            
        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        if self._global_pc is not None:
            self._fuse_distributions()
        else:
            self._global_pc = self.curr_pc
        return self.evaluations[max_idx][:2]

    def set_search_space(self, new_search_space: SearchSpace):
        """
            Set new search space and find out which variables have beend removed and which have been added.
        """
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
        self._curr_iter = 0
        self.evaluations = []

    def _fuse_distributions(self):
        # TODO: how to transfer knowledge, do we have to drop correlation knowledge?
        # 1. compute product distribution of current global PC and newly added dimensions.
        prod = self._cmopute_product_distribution(self._global_pc, self.search_space)
        # copy the PC as we do one marginalized version of it
        
        # 2. approximate second mixture component using rehearsal
        # sample from current pc
        if len(self.removed_vars) > 0:
            vars = list(self.search_space.get_search_space_definition().keys())
            cond = np.repeat([np.nan], self._global_conditional_approx_samples*len(vars)).reshape(self._global_conditional_approx_samples, len(vars))
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
        fs_r, ls_r = self.evaluations[0][1].val_performance, self.evaluations[-1][1].val_performance
        avg_optim_gain = np.mean(abs(fs_r - ls_r))
        w1 = sigmoid(avg_optim_gain)
        w2 = 1 - w1
        root = Sum([w1, w2], [prod, scd_pc])
        root.scope = scd_pc.scope
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

    def _build_dataset(self):
        """
            build a data matrix based on configuration and score pairs
        """
        cfgs, scores = [cfg for cfg, _, _ in self.evaluations], [score.val_performance for _, score, _ in self.evaluations]
        cfg_vecs = [self.transform.transform_configuration(cfg) for cfg in cfgs]
        features = np.array(cfg_vecs)
        y = -np.array(scores)
        return np.concatenate((features, y.reshape(-1, 1)), axis=1)

    def _learn_pc(self):
        # drop duplicates
        data = np.unique(self.data, axis=0)
        self.ctxt.add_domains(data)
        mu_hp = max(40, data.shape[0] // 100)
        self.curr_pc = learn_mspn(data, self.ctxt, min_instances_slice=mu_hp)

    def _sample(self):
        num_vars = len(list(self.search_space.get_search_space_definition().keys())) + 1
        cond_array = [np.nan] * num_vars
        # unconditional sampling
        num_unconditional_samples = int(self._warm_start_samples / 2)
        unconditional_samples = sample_instances(self.curr_pc, np.array([cond_array] * num_unconditional_samples), self._rand_state)

        # conditional sampling
        num_conditional_samples = int(self._warm_start_samples / 2)
        cond_array[-1] = self.data[:, -1].min()
        conditional_samples = sample_instances(self.curr_pc, np.array([cond_array] * num_conditional_samples), self._rand_state)
        samples = np.concatenate((unconditional_samples, conditional_samples), axis=0)
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