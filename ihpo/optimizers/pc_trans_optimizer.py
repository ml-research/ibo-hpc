from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..utils import ConfigurationNumericalTransform
import numpy as np
from ..utils.ibo import create_quantile_buckets, create_buckets, compute_bin_number
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
from spn.algorithms.EM import EM_optimization
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context, Product, Sum, assign_ids, get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import Categorical, CategoricalDictionary, Gaussian
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Marginalization import marginalize
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy
import warnings

warnings.filterwarnings("ignore")

class PCTransOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100, num_self_consistency_samplings=20, 
                num_samples=20, initial_samples=20, use_ei=False, num_ei_repeats=20,num_ei_samples=1000, 
                num_ei_variance_approx=10, transfer_deacy=0.3, decay_factor=0.9, pc_type='mspn', conditioning_value_quantile=1,
                 deactivate_transfer=False, explore_while_transfer=True, max_rel_ball_size=0.05, seed=0) -> None:
        super().__init__(search_space, objective)
        self.curr_pc = None
        self._num_iterations = iterations
        self._samples_per_iter = num_samples
        self._initial_samples = initial_samples
        self._rand_state = RandomState(seed)
        self._use_eic = use_ei
        self._deactivate_transfer = deactivate_transfer # if true, will act as PC optimizer
        self._explore_while_transfer = explore_while_transfer
        self._max_rel_ball_size = max_rel_ball_size
        # must be recorded for logging
        self._curr_iter = 0
        self._search_space_switched = False
        self._global_pc = None
        self._seen_dimensions = {name: (scope + 1) for scope, name in enumerate(search_space.get_search_space_definition())}
        self._seen_dimensions['_objective_'] = 0 # set objective's scope to 0
        self._task_pc_scopes_to_glob_pc_scopes = {}
        self._glob_pc_scopes_to_task_pc_scopes = {0: 0}
        self._search_space_definition_union = search_space.get_search_space_definition()
        self._first_and_last_sample = []
        self._num_ei_repeats = num_ei_repeats
        self._num_ei_samples = num_ei_samples
        self._num_ei_variance_approx = num_ei_variance_approx
        self._num_self_consistency_samplings = num_self_consistency_samplings
        self._pc_type = pc_type
        self._transfer_decay = transfer_deacy
        self._decay_factor = decay_factor
        self._conditioning_value_quantile = conditioning_value_quantile
        self._early_stopped_pc = None
        self.transform = ConfigurationNumericalTransform(search_space)
        self.scale_transform = StandardScaler()
        
    def optimize(self):
        """
            Fit a PC and use it to sample new promising configurations
        """
        for i in range(self._num_iterations):
            self._curr_iter = i
            print(f"Iteration {i+1}/{self._num_iterations}")
            if i % 10 == 0:
                #print(f"Iteration {i+1}/{self._num_iterations}")
                if i > 0:
                    val_performances = [e.val_performance for _, e, _ in self.evaluations]
                    print(f"Max performance: {max(val_performances)}")

            if self.curr_pc is None:
                # first task, no PC exists
                self._learn_init_pc()
            elif self._search_space_switched:
                # new task, use latest PC to get initial samples
                self._learn_trans_init_pc(k=1)
            else:
                # learn a new PC on current search space given fresh data
                self._learn_pc()
            configs = self._sample()
            self._evaluate(configs)

            #if i == int(self._num_iterations / 2):
            #    self._early_stopped_pc = deepcopy(self.curr_pc)

        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        if self._global_pc is not None and not self._deactivate_transfer:
            # after optimization, re-learn PC on scaled data to integrate into global_pc
            self._scale_data()
            self._learn_pc()
            self._fuse_distributions()
        else:
            self._global_pc = self.curr_pc
        final_config, final_res = self.evaluations[max_idx][:2]
        #self.curr_pc = self._early_stopped_pc
        return final_config, final_res

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
        self.search_space = new_search_space
        new_search_space_dimensions = set(new_search_space.get_search_space_definition().keys())
        new_search_space_dimensions = new_search_space_dimensions.union(set(['_objective_']))
        
        # find new variables and unused variables
        seen_vars = set(self._seen_dimensions.keys())
        self.added_vars = list(new_search_space_dimensions.difference(seen_vars))
        self.removed_vars = list(seen_vars.difference(new_search_space_dimensions))

        # set configuration transformation
        self.transform = ConfigurationNumericalTransform(new_search_space)

        # set new context for PC
        added_dimensions = {}
        max_scope = max(list(self._seen_dimensions.values()))
        for i, name in enumerate(self.added_vars):
            added_dimensions[name] = max_scope + (i + 1)
        self._seen_dimensions = {**self._seen_dimensions, **added_dimensions}
        self._search_space_definition_union = {**self._search_space_definition_union, **new_search_space.get_search_space_definition()}

        # adapt global PC to new search sapce and set as current PC
        if self._global_pc is not None:
            intersect = list(seen_vars.intersection(new_search_space_dimensions)) + ['_objective_']
            intersect_scope = [self._seen_dimensions[n] for n in intersect]
            marg_pc = self._marginalize(self._global_pc, intersect_scope)
            self.curr_pc = self._cmopute_product_distribution(marg_pc, self.added_vars, new_search_space.get_search_space_definition())

        self._construct_search_space_metadata()
        if self.curr_pc is not None:
            self._search_space_switched = True
        self.evaluations = []

    def _scale_data(self):
        objective_idx = self._seen_dimensions['_objective_']
        score = self.data[:, objective_idx].reshape(-1, 1)
        scaled_score = self.scale_transform.fit_transform(score)
        self.data[:, objective_idx] = scaled_score.flatten()

    def _fuse_distributions(self):
        """
            Given the PC learned on the current task (current_pc) and the PC that accumulated knowledge over all t-1 former tasks,
            fuse the current PC with the global PC s.t. knowledge from both distributions is entailed in the new PC.

            There are three cases to consider:
            1. We have a homogeneous search space. Then, fusing is only building a mixture over the current PC and the global PC
            2. The current search space contains strictly more variables than former spaces. Then, a product node has to be added in the global
                PC since the current PC already was extended that way when preparing it for the current task.
                Intuitively, the current PC then contains dependence information between variables of the current search space while the global
                PC contains dependence information between variables of all former tasks. This information gets merged in the mixture.
            3. There are variables that are entailed in former search spaces but not in the current one AND there are variables that are introduced
                by the current search space. Similarly to 2., we now have to add a product node to the global PC to cover the variables introduced
                by the current search space. Additionally, we have to add a product node to the current PC to cover the variables that do not exist
                in the current search space but in the former spaces. 
        """

        # 0. make task-specific PC scope compatible with global PC scope
        for node in get_nodes_by_type(self.curr_pc):
            new_scope = []
            for s in node.scope:
                new_scope.append(self._task_pc_scopes_to_glob_pc_scopes[s])
            node.scope = new_scope

        # 1. Do we have new variables from current space? Then add them in global PC
        prod_global_pc = self._cmopute_product_distribution(self._global_pc, self.added_vars, self._search_space_definition_union)

        # 2. Do we have variables that are not in current space but in older ones? Then add to current PC
        prod_curr_pc = self._cmopute_product_distribution(self.curr_pc, self.removed_vars, self._search_space_definition_union)

        # 3. merge distributions into mixture
        fs, ls = self._first_and_last_sample
        fs_r, ls_r = fs[:, -1], ls[:, -1]
        # compute gain we got from optimizing
        avg_optim_gain = np.mean(abs(fs_r - ls_r))
        # weight mixture components such that the one that's expected to yield
        # better configurations is sampled more often
        w1 = sigmoid(avg_optim_gain)
        w2 = 1 - w1
        root = Sum([0.5, 0.5], [prod_global_pc, prod_curr_pc])
        root.scope = prod_curr_pc.scope
        root = assign_ids(root)
        #EM_optimization(root, self.data, 2)
        self._global_pc = root

    def _marginalize(self, pc, keep):
        marg_pc = marginalize(pc, keep)
        return marg_pc
    
    def _cmopute_product_distribution(self, pc, new_vars, ssd: dict):
        """Add product nodes for all non-existent dimensions in PC.

        Args:
            pc (SPN): PC object
            new_vars (list): List of variables not existing in PC
            ssd (dict): Search space description for all variables in new_vars.

        Returns:
            SPN: PC with extended scope
        """
        if len(new_vars) == 0:
            return pc
        scopes = []

        # add product distribution
        prod_children = []
        for name in new_vars:
            scope = self._seen_dimensions[name]
            scopes.append(scope)
            if ssd[name]['type'] == MetaType.DISCRETE:
                vals = ssd[name]['allowed']
                p = [1/len(vals)] * len(vals)
                node = Categorical(p=p, scope=scope)
            elif ssd[name]['type'] == MetaType.REAL:
                # since Uniform distribution from SPFlow is not documented, we approximate it by samples
                min_, max_ = ssd[name]['min'], ssd[name]['max']
                choices = np.random.uniform(min_, max_, size=1000)
                probs = [1/1000] * 1000
                prob_dict = {v: p for p, v in zip(probs, choices)}
                node = CategoricalDictionary(p=prob_dict, scope=scope)
            prod_children.append(node)

        prod_node = Product([pc] + prod_children)
        prod_node.scope = pc.scope + scopes
        prod_node = assign_ids(prod_node)
        return prod_node

    def _build_dataset(self, configs, scores):
        """
            build a data matrix based on configuration and score pairs
        """
        data_matrix = []
        config_col_order = self._get_config_column_order()
        for cfg, score in zip(configs, scores):
            # transform configurations into vector-form and encode categoricals by numbers
            features = [score] + self.transform.transform_configuration(cfg, config_col_order)
            data_matrix.append(features)
        data = np.array(data_matrix)
        return data
    
    def _learn_init_pc(self):
        """
            First, generate a set of random samples from the search space and evaluate them.
            Then, build a dataset and learn a PC over the joint space of hyperparameters and evaluation score.
        """
        configs = self.search_space.sample(size=self._initial_samples)
        scores = [self.objective(cfg) for cfg in configs]
        val_scores = [s.val_performance for s in scores]
        self.data = self._build_dataset(configs, val_scores)
        self._learn_pc()

    def _learn_pc(self):
        """
            Learn a MSPN over the joint space of hyperparameters and evaluations.
        """
        if self._pc_type == 'parametric':
            self._learn_pc_parametric()
        elif self._pc_type == 'quantile':
            num_buckets = compute_bin_number(self.data.shape[0])
            self._learn_pc_quantile(num_buckets, split_mode='linear', q_max=0.97)
        else:
            self._construct_search_space_metadata()
            self.ctxt.add_domains(self.data)
            mu_hp = max(5, self.data.shape[0] // 100)
            try:
                self.curr_pc = learn_mspn(self.data, self.ctxt, min_instances_slice=15, threshold=0.3, n_clusters=2)
            except ValueError:
                # sometimes learning fails due to convergence failure of RDC. Try again with less restrictions (model will be more inacurrate)
                print("PC FIT FAILED...RETRY")
                self.curr_pc = learn_mspn(self.data, self.ctxt, min_instances_slice=40, threshold=0.1, n_clusters=2)

    def _learn_pc_parametric(self):
        """
            Learn a parametric SPN (Gaussian leaves).
            This requires all hyperparameters to be continuous.
        """
        params = [Gaussian] * (self.data.shape[1])
        meta_types = [MetaType.REAL] * (self.data.shape[1])
        ctxt = Context(meta_types=meta_types, parametric_types=params)
        X_joint = self.data
        ctxt.add_domains(X_joint)
        self.curr_pc = learn_parametric(X_joint, ctxt, min_instances_slice=8, threshold=0.1)

    def _learn_pc_quantile(self, buckets=20, split_mode='linear', min_samples_per_bucket=10, q_max=0.99):
        """
            Learn a PC using the quantile approach: First, we bucketize the evaluation scores.
            Depending on 'split_mode', this happens by splitting the sorted array of evaluation scores
            in equally sized slices, by performing recursive splitting of the evaluation space or both.

            # Equal Splits
            If we split equally, the evaluation scores get sorted and the is split s.t. each bucket contains
            the same number of samples. Since the configurations remain assigned to their score, we split
            the data space into bad, medium and good samples. For each split (possibly more than 3 as illustrated here),
            we fit a PC and form a mixture over all PCs in the end.

            # Recursive Splits
            We start by computing the 50%-quantile, split the space into "good" and "bad", pick the "good" part of the 
            data and repeat this procedure until the smallest bucket contains 'min_samples_per_bucket' samples. 
            Then, for each bucket a PC is learned and a mixture of PCs is formed as a final model.

            # Equal Splits + Recursive Splits
            We first apply recursive splittint of the data space and then perform equal splitting with all 
            "bad" parts of the search space. Again, a mixture of PCs is formed in the end to obtain the full model.

        Args:
            buckets (int, optional): Number of buckets to split the data into. Defaults to 20.
            split_mode (str, optional): Mode of splitting. Can be 'linear', 'binary_tree' or 'bt+lin'. Defaults to 'linear'.
            min_samples_per_bucket (int, optional): Minimum samples required per bucket. Defaults to 10.
            q_max (float, optional): Maximum quantile used if split_mode is 'binary_trees' or 'bt+lin'. Defaults to 0.99.
        """
        self._construct_search_space_metadata()
        self.ctxt.add_domains(self.data)
        y = self.data[:, -1]
        idx = np.argsort(y).flatten()
        if split_mode == 'linear':
            indices = np.array_split(idx, buckets)
        elif split_mode == 'binary_tree':
            _, indices = create_buckets(y[idx], idx, min_samples=min_samples_per_bucket)
        elif split_mode == 'bt+lin':
            _, indices = create_buckets(y[idx], idx, min_samples=min_samples_per_bucket)
            first_idx = indices[0]
            first_indices = np.array_split(first_idx, buckets)
            indices = first_indices + indices[1:]
        elif split_mode == 'quantile':
            _, indices = create_quantile_buckets(y, q_max=q_max)
        spns = []
        weights = []
        for idx_chunk in indices:
            subset = self.data[idx_chunk]
            weights.append(len(idx_chunk))
            try:
                spn = learn_mspn(subset, self.ctxt, min_instances_slice=10, threshold=0.4, n_clusters=2)
            except ValueError:
                # sometimes learning fails due convergence failure of RDC. Try again with less restrictions
                spn = learn_mspn(subset, self.ctxt, min_instances_slice=40, threshold=0.5, n_clusters=2)
            spns.append(spn)

        sum = Sum()
        scope = spns[0].scope
        sum.scope = scope
        weights = np.array(weights) / np.sum(weights)
        sum.weights = weights
        sum.children = spns
        spn = assign_ids(sum)
        self.curr_pc = spn

    def _learn_trans_init_pc(self, k=5):
        assert int(round(self._samples_per_iter / k)) == int(self._samples_per_iter / k), 'k must divide samples per iteration'

        config_col_order = self._get_config_column_order()
        
        # 1. obtain samples from curr_pc
        samples = self._sample(is_transfer_iter=True)

        # 1.5 use samples to init ball
        #samples = sample_from_ball(samples, 0.5, samples.shape[0], samples.shape[1])        

        # 2. evaluate sampled configurations
        configs = []
        for sample in samples:
            d = self.transform.inv_transform_configuration(sample, config_col_order)
            configs.append(d)

        #if self._explore_while_transfer:
        #    # replace some of the samples from PC by random samples to foster exploration in the beginning
        #    sample_size = int(len(samples) / 2)
        #    random_samples = self.search_space.sample(size=sample_size)
        #    replace_idx = np.random.randint(0, len(samples), size=sample_size)
        #    for s, idx in zip(random_samples, replace_idx):
        #        configs[idx] = s

        scores = [self.objective(cfg) for cfg in configs]
        val_scores = np.array([s.val_performance for s in scores])
        self.data = np.column_stack((val_scores.reshape(-1, 1), samples))

        # 3. fit PC
        self._learn_pc()
        self._search_space_switched = False # reset switch of search space indicator

    def _get_conditioning_value(self, quantile=1):
        """
            Only used for ablation studies to analyze importance of conditioning on the best value.
            If optimizer is running in normal mode, set self._conditioning_value_quantile=1.
        """
        objective_scope = self._seen_dimensions['_objective_']
        if quantile == 1:
            return self.data[:, objective_scope].max()
        else:
            return np.quantile(self.data[:, objective_scope], quantile)

    def _sample(self, is_transfer_iter=False):
        """
            Sample from the PC to obtain the next configuration(s) to test.
            Depending on whether `use_ei` is True or False, the next configurations
            are selected using the EI acquisition function or via self-consistent
            conditional sampling.

            # EI
            If EI is used, 'num_ei_repeats'*'num_ei_samples' random samples are drawn from the search space
            and the best 'num_ei_repeats' ones are kept and tested.

            # Self-consistent Conditional Sampling
            If the self-consistent conditional sampling is used, 'num_self_consistency_samplings' rounds of 
            conditional sampling are performed as follows: In each round, 'num_samples' configurations are 
            drawn from the PC by conditioning the PC on the best evaluation score obtained so far. Then, each
            configuration is fed back to the PC which in turn predicts the expected evaluation score each
            sampled configuration will achieve. In each sampling round, we keep the configuration with the
            highest expected evaluation score.

        Args:
            is_transfer_iter (bool, optional): Indicates whether we're in an initial transfer iteration. Defaults to False.

        Returns:
            List: Set of configurations tested next.
        """
        ssd = list(self.search_space.get_search_space_definition().keys()) + ['_objective_']
        if is_transfer_iter:
            num_vars = len(self._seen_dimensions)
            hp_idx = sorted([idx for name, idx in self._seen_dimensions.items() if name in ssd])
        else:
            num_vars = len(ssd)

        cond_array = np.array([np.nan] * self._samples_per_iter * num_vars).reshape(-1, num_vars)
        objective_scope = self._seen_dimensions['_objective_']
        cond_array[:, objective_scope] = self._get_conditioning_value()

        # EI sampling
        if self._use_eic:
            sampled_configs = self._ei()
        # Conditional sampling with self-reensurance
        else:
            best_configs = []
            for i in range(self._num_self_consistency_samplings):
                samples_ = sample_instances(self.curr_pc, np.array(cond_array), np.random.RandomState(i))
                re_ensurance_cond = np.copy(samples_)
                re_ensurance_cond[:, objective_scope] = np.nan
                y_pred = sample_instances(self.curr_pc, re_ensurance_cond, np.random.RandomState(i))[:, objective_scope].flatten()
                idx = np.argmax(y_pred).flatten()[0]
                selected_samples = samples_[idx]
                if is_transfer_iter:
                    best_configs.append(selected_samples[hp_idx])
                else:
                    best_configs.append(selected_samples)
            samples = np.array(best_configs)
            y_ = samples[:, objective_scope]
            idx = np.argsort(y_).flatten()[:self._samples_per_iter]
            samples = samples[idx]
            cfg_col_idx = np.array([i for i in range(samples.shape[1]) if i != objective_scope])
            sampled_configs = samples[:, cfg_col_idx]

        if self._global_pc is not None:
            col_order = self._get_config_column_order()
            rnd_prob = self._transfer_decay * self._decay_factor**self._curr_iter
            sample_size = int(round(rnd_prob*self._samples_per_iter))
            print(f"Sampling {sample_size} random samples.")
            random_samples = self.search_space.sample(size=sample_size)
            random_samples = np.array([self.transform.transform_configuration(cfg, col_order) for cfg in random_samples])
            replace_idx = np.random.randint(0, len(sampled_configs), size=sample_size)
            for s, idx in zip(random_samples, replace_idx):
                sampled_configs[idx] = s

        var = np.var(sampled_configs, axis=0)
        #r = np.array([min(1/dim_var, self.ball_radius[i]) for i, dim_var in enumerate(var)])
        r = self.ball_radius

        sampled_configs = sample_from_ball(sampled_configs, self.search_space, r, sampled_configs.shape[0])

        if is_transfer_iter:
            cond_array = np.array([np.nan] * self._samples_per_iter * num_vars).reshape(-1, num_vars)
            samples_ = sample_instances(self.curr_pc, np.array(cond_array), np.random.RandomState(self._curr_iter))
            sampled_configs = np.concatenate((sampled_configs, samples_[:, cfg_col_idx]))

        # ensure that sampled values are in allowed range
        for k in self.search_space.get_search_space_definition().keys():
            v = self.search_space.get_search_space_definition()[k]
            i = self._glob_pc_scopes_to_task_pc_scopes[self._seen_dimensions[k] - 1] # -1 as we removed score dimension above
            if v['dtype'] == 'float':
                for j in range(sampled_configs.shape[0]):
                    sampled_configs[j, i] = max(sampled_configs[j, i], v['min'])
                    sampled_configs[j, i] = min(sampled_configs[j, i], v['max'])
        return sampled_configs

    def _evaluate(self, sampled_configs):
        """
            For all sampled configurations, invoke the evaluation procedure.
            The results are added to the dataset used to fit the PC.

        Args:
            sampled_configs (_type_): Set of sampled configurations
        """
        evaled_samples = []
        config_col_order = self._get_config_column_order()
        for i in range(len(sampled_configs)):
            sample = list(sampled_configs[i])
            # if already sampled, skip
            #if np.any(np.all(self.data[:, :-1] == np.array(sample), axis=1)):
            #    continue
            config_dict = self.transform.inv_transform_configuration(sample, config_col_order)
            evaluation = self.objective(config_dict)
            if evaluation is not None:
                self.evaluations.append((config_dict, evaluation, self._curr_iter))
                sample = [evaluation.val_performance] + sample
                evaled_samples.append(sample)
        evaled_samples = np.array(evaled_samples)
        if evaled_samples.shape[0] > 0:
            # may be that we have seen all sampled configs already
            self.data = np.concatenate((self.data, evaled_samples))
        if len(self._first_and_last_sample) in [0, 1]:
            self._first_and_last_sample.append(evaled_samples)
        else:
            self._first_and_last_sample[-1] = evaled_samples

    def _ei(self):
        best_samples = []
        objective_idx = self._seen_dimensions['_objective_']
        config_col_order = self._get_config_column_order()
        for j in range(self._num_ei_repeats):
            eic_samples = self.search_space.sample(size=self._num_ei_samples)
            eic_samples = np.array([self.transform.transform_configuration(cfg, config_col_order) for cfg in eic_samples])
            nan_column = np.full((eic_samples.shape[0], 1), np.nan)
            cond_matrix = np.hstack((nan_column, eic_samples))
            pred_mat = []
            for i in range(self._num_ei_variance_approx):
                preds = sample_instances(self.curr_pc, cond_matrix, np.random.RandomState(i))[:, objective_idx]
                pred_mat.append(preds)
            pred_mat = np.array(pred_mat)
            mean_preds = np.mean(pred_mat, axis=0)
            mean_std = np.std(pred_mat, axis=0)
            improve = self._get_conditioning_value() - 0.01 - mean_preds
            scaled = improve / mean_std
            cdf = scipy.stats.norm.cdf(scaled)
            pdf = scipy.stats.norm.pdf(scaled)
            exploit = improve * cdf
            explore = mean_std * pdf
            best = np.argsort(explore + exploit)
            best_samples.append(eic_samples[best[0]])
        return np.array(best_samples)
    
    def _construct_search_space_metadata(self):
        search_space_definition = self.search_space.get_search_space_definition()
        meta_types = [MetaType.REAL] + [val['type'] for val in search_space_definition.values()]
        self.hyperparam_names = list(search_space_definition.keys())
        self.ctxt = Context(meta_types=meta_types)

        # set mapping between task-specific PC and global PC
        self._task_pc_scopes_to_glob_pc_scopes = {0: 0}
        for task_pc_scope_idx, hp_name in enumerate(search_space_definition.keys()):
            global_pc_scope_idx = self._seen_dimensions[hp_name]
            self._task_pc_scopes_to_glob_pc_scopes[task_pc_scope_idx + 1] = global_pc_scope_idx   
            self._glob_pc_scopes_to_task_pc_scopes[global_pc_scope_idx] = task_pc_scope_idx + 1 # +1 as we have to include score dimension
        
        # set radius for sampling
        self.ball_radius = []
        for idx, (name, hp_def) in enumerate(search_space_definition.items()):
            if hp_def['dtype'] == 'float':
                if isinstance(self._max_rel_ball_size, list):
                    rel_rad = self._max_rel_ball_size[idx]
                else:
                    rel_rad = self._max_rel_ball_size
                
                if hp_def['is_log']:
                    r = (np.log(hp_def['max']) - np.log(hp_def['min'])) * rel_rad
                else:
                    r = (hp_def['max'] - hp_def['min']) * rel_rad
                self.ball_radius.append(r)
            else:
                # if discrete HP, interprete the relative max ball size as a probability to sample randomly from the dimension
                if isinstance(self._max_rel_ball_size, list):
                    self.ball_radius.append(self._max_rel_ball_size[idx])
                else:
                    self.ball_radius.append(self._max_rel_ball_size)

    def _get_config_column_order(self):
        """Get a list of hyperparameter names s.t. its order corresponds to the order of columns/RVs expected by the PC.

        Returns:
            list: List of HPs of the current search space, ordered by PC expectation.
        """
        curr_vars = list(self.search_space.get_search_space_definition().keys())
        # NOTE: Do not change implementation of this! The order of the list must be preserved!
        cols = [c for c in self._seen_dimensions.keys() if c in curr_vars]
        return cols     
        
    @property
    def history(self):
        return self.evaluations
    
def sample_from_ball(center, search_space: SearchSpace, radius, n_samples):
    """
    Sample random points uniformly from a ball around a given center.
    
    Parameters:
        center (array-like): Coordinates of the center of the ball.
        search_space (SearchSpace): Search space of the current HPO task.
        radius (float): Radius of the ball.
        n_samples (int): Number of points to sample.

    Returns:
        np.ndarray: Array of shape (n_samples, dimension) with sampled points.
    """
    if isinstance(radius, list):
        radius = np.array(radius)
    points = []

    search_space_def = search_space.get_search_space_definition()

    # only apply ball sampling to real dimensions
    real_dims = [i for i, v in enumerate(search_space_def.values()) if v['dtype'] == 'float']
    other_dims = [(name, i) for i, (name, v) in enumerate(search_space_def.items()) if v['dtype'] != 'float']

    for i in range(n_samples):
        if len(real_dims) > 0:
            # Generate a random direction (unit vector) in `dimension` space
            direction = np.random.normal(size=len(real_dims))
            direction /= np.linalg.norm(direction)
            
            # Generate a random distance from the center, scaled to fit inside the radius
            distance = np.random.uniform(0, 1) ** (1 / len(real_dims)) * radius[real_dims]
            
            # Compute the point
            center[i, real_dims] = center[i, real_dims] + direction * distance
        for name, j in other_dims:
            coin = np.random.choice([0, 1], p=[1-radius[j], radius[j]])
            if coin == 1:
                choices = np.arange(0, len(search_space_def[name]['allowed']))
                rnd_cfg = np.random.choice(choices)
                center[i, j] = rnd_cfg
        points.append(center[i])
    
    return np.array(points)