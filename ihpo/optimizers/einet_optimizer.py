from ..utils.einet.layers.distributions.piecewise_linear import PiecewiseLinear
from ..utils.einet.dist import Domain, DataType
from ..utils.einet.learning import train_einet, train_shift_and_mixture_model, train_second_order_mixture_model
from ..utils.einet.regression import MixtureEinetRegressionEngine, EvolutionaryOptimizationEinetRegressionEngine
from ..utils.einet.utils import init_einet_domains
from ..utils import ConfigurationNumericalTransform, load_task_files
from ..utils.optimization import optimize_evolutionary
from ..utils.acquisitions import EI
from ..consts.dtypes import MetaType
from .optimizer import Optimizer
from sklearn.metrics import r2_score
import numpy as np
import torch

class EinetOptimizer(Optimizer):

    def __init__(self, search_space, objective, seed=0, iterations=100, init_samples=10, samples_per_iter=5, 
                 max_num_sums=25, max_num_leaves=20, max_num_repitions=20, prior_hpo_run_files=None, prior_search_spaces=None,
                 num_transfer_samples=100, num_transfer_functions=100, num_warmstart_samples=10,
                 hierarchical_mixture=True, transfer_learning_epochs=100, use_ei=False, num_ei_samples=1000):
        super().__init__(search_space, objective, seed)
        self.transform = ConfigurationNumericalTransform(search_space)
        self.domains = init_einet_domains(search_space)
        self._iterations = iterations
        self._init_samples = init_samples
        self._samples_per_iter = samples_per_iter
        self._max_num_sums = max_num_sums
        self._max_num_leaves = max_num_leaves
        self._max_num_repititions = max_num_repitions
        self._prior_hpo_run_files = prior_hpo_run_files
        self._prior_search_spaces = prior_search_spaces
        self._num_transfer_functions = num_transfer_functions
        self._num_transfer_samples = num_transfer_samples
        self._num_warmstart_samples = num_warmstart_samples
        self._hierarchical_mixture = hierarchical_mixture
        self._transfer_learning_epochs = transfer_learning_epochs
        self._use_ei = use_ei
        self._num_ei_samples = num_ei_samples
        self._history = []
        self.einet = None
        self.model_history = [] # important for transfer learning. contains previous models.
        self.mixture_weights = None
        self.second_order_mixture_weights = None # will only be set if hierarchical_mixture=True

        if prior_hpo_run_files is not None:
            self._pretrain()

    def optimize(self):

        # Step 0: Sample and evaluate random initial samples
        samples = self._sample_initial_samples()
        self.evaluations = [self.objective(cfg) for cfg in samples]
        self._history += [(cfg, e, i) for i, (cfg, e) in enumerate(zip(samples, self.evaluations))]
        transformed_samples = np.array([self.transform.transform_configuration(cfg) for cfg in samples]).astype(np.float32)
        eval_score = np.array([ev.val_performance for ev in self.evaluations]).astype(np.float32)
        self.data = torch.from_numpy(np.hstack((transformed_samples, eval_score.reshape(-1, 1))))

        for i in range(self._iterations):

            # Step 1: learn einet on given data
            self.einet = train_einet(self.data.unsqueeze(1), self.domains, max_num_repitions=self._max_num_repititions,
                                     max_num_leaves=self._max_num_leaves, max_num_sums=self._max_num_sums, log_loss=False)
            
            # Step 2: perform knowledge transfer via optimization
            # TODO: do we need to fit this every time?
            if len(self.model_history) > 0:
                self._train_history_mixture()

            # Step 3: sample new configurations given best obtained score
            # TODO: Sometimes samples contain nan values when using conditional sampling, why?
            # TODO: when doing HTL, suggestions are really bad. Why?
            samples = self._suggest_next_config()

            # Step 4: evaluate configuration(s) and update dataset
            new_evaluations = [self.objective(self.transform.inv_transform_configuration(sample.numpy())) for sample in samples]
            self._history += [(self.transform.inv_transform_configuration(cfg.numpy()), e, i) for i, (cfg, e) in enumerate(zip(samples, new_evaluations))]
            self.evaluations += new_evaluations
            new_evaluations = torch.from_numpy(
                np.array([ev.val_performance for ev in new_evaluations]).astype(np.float32)
            )
            
            nan_col = torch.full((samples.shape[0], 1), float('nan'))
            samples = torch.cat([samples, nan_col], dim=1)
            samples[:, -1] = new_evaluations
            self.data = torch.cat((self.data, samples))

            print(self.data[:, -1].max())
            #self._log_r2()

        max_idx = torch.argmax(self.data[:, -1]).item()
        cfg = self.evaluations[max_idx]
        return cfg, self.data[max_idx, -1].item()

    def set_search_space(self, search_space):
        if self.einet is not None:
            self.model_history.append((self.einet, self.domains, self.search_space, 
                                       self.transform, self.data[:, -1].max()))
        self.search_space = search_space
        self.transform = ConfigurationNumericalTransform(search_space)
        self.domains = init_einet_domains(search_space)
        self._history = []

    def _pretrain(self):
        """
            In case the user specified pre-existing HPO run logs, load them and pre-train einets on the data
        """ 
        assert len(self._prior_hpo_run_files) == len(self._prior_search_spaces), 'Require all prior search spaces and logs for pretraining.'

        for search_space, file in zip(self._prior_search_spaces, self._prior_hpo_run_files):
            transform = ConfigurationNumericalTransform(search_space)
            data = load_task_files([file])
            transformed_data = np.array([transform.transform_configuration(cfg) for cfg in data.to_dict(orient='records')])
            domains = init_einet_domains(search_space)
            einet = train_einet(torch.from_numpy(transformed_data), domains)
            self.model_history.append((einet, domains, search_space, transform, transformed_data[:, -1].max()))
            

    def _train_history_mixture(self):
        """
            Train the mixture and shift parameters of the einets in the history.

            If self._hierarchical mixture is True, we exclude the current einet from the optimization and 
            learn a separate set of weights determining the influence of the current einet and the mixture
            (this follows the TransBO approach).

            If self._hierarchical mixture is False, the current einet is considered directly in the optimization.
            This might lead to ignoring the einets in the historical, i.e., reduced knowledge transfer.
        """
        if self._hierarchical_mixture:
            einets = [einet for einet, _, _, _, _ in self.model_history]
            search_spaces = [space for _, _, space, _, _ in self.model_history]
            self.mixture_weights = torch.nn.Parameter(torch.rand(len(einets)))
            self.mixture_weights, einets = train_shift_and_mixture_model(einets, search_spaces, self.search_space, self.data, 
                                                                         self.mixture_weights, self._transfer_learning_epochs, log_loss=True)

            self.second_order_mixture_weights = torch.nn.Parameter(torch.rand(2))
            self.second_order_mixture_weights = train_second_order_mixture_model(einets, self.einet, search_spaces, self.search_space, self.data, self.mixture_weights,
                                                                                 self.second_order_mixture_weights, self._transfer_learning_epochs)

        else:
            einets = [einet for einet, _, _, _, _ in self.model_history] + [self.einet]
            search_spaces = [space for _, _, space, _, _ in self.model_history]
            self.mixture_weights = torch.nn.Parameter(torch.rand(len(einets)))
            self.mixture_weights, einets = train_shift_and_mixture_model(einets, search_spaces, self.search_space, self.data, 
                                                                         self.mixture_weights, self._transfer_learning_epochs)
            
    def _sample_initial_samples(self):
        """
            Sample the first initial samples for a new task.
            If there is no model history yet, simply sample randomly from search space.
            Else perform knowledge transfer.
        """
        if len(self.model_history) == 0:
            return self.search_space.sample(size=self._init_samples)
        else:
            # get initial samples based on previous models to transfer knowledge.
            # Step 1: Sample HPs given best value for each task (minimizes Eq. 12 from FSBO).
            per_task_warm_start_configs = self._sample_init_values_from_each_task()

            # Step 2: generate random mixture weights and shift parameters, i.e., random functions from previous surrogates
            disc_shift_params, cont_shift_params, mixture_weights = self._init_random_shift_and_mixture_parameters()

            # Step 3: optimize all random functions and obtain the best warm-start values.
            #         Keep generated points for bootstrapping Step 4 (optimization-driven transfer of models).
            #         Also, do prediction for per_task_warm_start_configs, only the most robust ones will be kept
            per_task_configs_score_preds = []
            evolutionary_cfgs = []
            rnd_function_idx = np.random.randint(0, mixture_weights.shape[1], size=self._num_warmstart_samples // 2)
            for idx in rnd_function_idx:
                regression_engine = EvolutionaryOptimizationEinetRegressionEngine(self.model_history, self.search_space, 
                                                        mixture_weights, cont_shift_params, disc_shift_params, idx)
                configs, s = optimize_evolutionary(self.search_space, regression_engine.predict, 1)
                evolutionary_cfgs.append(configs[0])

                pt_configs_score_preds = []
                for cfg in per_task_warm_start_configs:
                    score_pred = regression_engine.predict(list(cfg.values()))
                    pt_configs_score_preds.append(score_pred.item())
                per_task_configs_score_preds.append(np.array(pt_configs_score_preds))

            per_task_configs_score_preds = np.stack(per_task_configs_score_preds, axis=1)
            rank_matrix = per_task_configs_score_preds.argsort(axis=0).argsort(axis=0)
            avg_rank_per_cfg = rank_matrix.mean(axis=1)
            # keep best
            sorted_avg_ranks = np.argsort(avg_rank_per_cfg)[:self._num_warmstart_samples // 2]
            per_task_warm_start_configs = [per_task_warm_start_configs[idx] for idx in sorted_avg_ranks]
            return per_task_warm_start_configs + evolutionary_cfgs
        
    def _suggest_next_config(self):
        """
            Suggest the next configuration to test.

            If self._use_ei is True, we use standard EI-based configuration selection.

            If self._use_ei is False, we use conditional sampling from PC(s)
        """ 
        if not self._use_ei:
            # NOTE: This approach doesn't work as models can be defined over different spaces
            # TODO: Implement extension of einsums to sample missing dimensions at random
            if len(self.model_history) > 0:
                models = [einet for einet, _, _, _, _ in self.model_history] + [self.einet]
                sampled_model = None
                samples = []
                # transfer learning case
                for s in range(self._samples_per_iter):
                    if self.second_order_mixture_weights is not None:
                        w = torch.softmax(self.second_order_mixture_weights, dim=0).detach().numpy()
                        coin = np.random.choice([0, 1], size=1, p=w)[0]
                        if coin == 1:
                            sampled_model = self.einet
                        else:
                            models = models[:-1]
                    
                    if sampled_model is None:
                        w = torch.softmax(self.mixture_weights, dim=0).detach().numpy()
                        model_idx = np.random.choice(list(range(len(models))), size=1, p=w)[0]
                        sampled_model = models[model_idx]

                    best_sample = self.data[self.data[:, -1] == self.data[:, -1].max()]
                    condition = best_sample.squeeze()
                    nan_idx = list(range(0, self.data.shape[1] - 1))
                    condition[nan_idx] = torch.nan
                    sample = sampled_model.sample(evidence=condition.unsqueeze(0).unsqueeze(0)).squeeze()[nan_idx]
                    samples.append(sample)
                samples = torch.cat(samples, dim=0)
            else:
                # standard HPO case
                sampled_model = self.einet

                best_sample = self.data[self.data[:, -1] == self.data[:, -1].max()]
                condition = best_sample.repeat(self._samples_per_iter, 1)
                nan_idx = list(range(0, self.data.shape[1] - 1))
                condition[:, nan_idx] = torch.nan
                samples = sampled_model.sample(evidence=condition.unsqueeze(1)).squeeze()[:, nan_idx]
        else:
            if len(self.model_history) > 0:
                # if we have previous models, take them into account
                if self._hierarchical_mixture:
                    regression_engine = MixtureEinetRegressionEngine(self.model_history, self.search_space, self.mixture_weights)
                    scd_order_mixture_weights = torch.softmax(self.second_order_mixture_weights, dim=0)
                else:
                    model_history = self.model_history + [(self.einet, self.domains, self.search_space, 
                                                           self.transform, self.data[:, -1].max())]
                    regression_engine = MixtureEinetRegressionEngine(model_history, self.search_space, self.mixture_weights)

            samples = []
            for _ in range(self._samples_per_iter):
                rnd_samples = np.array([self.transform.transform_configuration(cfg) for cfg in self.search_space.sample(size=self._num_ei_samples)])
                ext_rnd_samples = torch.from_numpy(rnd_samples).repeat_interleave(10, dim=0)

                if len(self.model_history) > 0:
                    # if previous models are available, use them
                    pred = regression_engine.predict(ext_rnd_samples, False)

                    if self._hierarchical_mixture:
                        nan_col = torch.full((ext_rnd_samples.shape[0], 1), float('nan'))
                        ext_rnd_samples = torch.cat([ext_rnd_samples, nan_col], dim=1)
                        einet_pred = self.einet.sample(evidence=ext_rnd_samples).squeeze()[:, -1]
                        pred = scd_order_mixture_weights[0] * pred + scd_order_mixture_weights[1] * einet_pred
                else:
                    nan_col = torch.full((ext_rnd_samples.shape[0], 1), float('nan'))
                    ext_rnd_samples = torch.cat([ext_rnd_samples, nan_col], dim=1)
                    pred = self.einet.sample(evidence=ext_rnd_samples).squeeze()[:, -1]
                
                mean_pred = pred.view(rnd_samples.shape[0], 10).mean(dim=1).squeeze().detach()
                std_pred = pred.view(rnd_samples.shape[0], 10).std(dim=1).squeeze().detach()

                incumbent = self.data[:, -1].max().numpy()
                #print(std_pred.numpy())
                # TODO: Check out nan values in EI!
                ei = EI(incumbent, mean_pred.numpy(), std_pred.numpy())
                max_idx = np.argmax(ei)
                samples.append(rnd_samples[max_idx])

            samples = torch.from_numpy(np.array(samples))
        return samples


    def _sample_init_values_from_each_task(self):
        """
            For each surrogate (one per task), sample high performing configurations.
            If current search space entails dimensions not entailed in former model(s), we sample from the uniform distribution
            for this dimension.
        """
        intial_configs = []
        for einet, domain, search_space, transform, best_score in self.model_history:
            space_dim = len(domain)
            cond_arr = torch.full((self._num_transfer_samples, space_dim), float('nan')).to(torch.float32)
            cond_arr[:, -1] = best_score
            samples = einet.sample(evidence=cond_arr)

            # self-ensure that the sampled values are good
            #samples[:, -1] = torch.nan
            #samples = einet.sample(evidence=samples)

            # set configuration with sampled values for all dimensions that exist in current and former search space.
            # All other dimensions are set randomly.
            config_suggestions = self.search_space.sample(size=samples.shape[0])
            sampled_configs = [transform.inv_transform_configuration(s.squeeze().numpy()[:-1]) for s in samples]

            for key in search_space.get_search_space_definition().keys():
                if key in config_suggestions[0]:
                    for idx in range(len(sampled_configs)):
                        config_suggestions[idx][key] = sampled_configs[idx][key]

            intial_configs += config_suggestions
        return intial_configs


    def _init_random_shift_and_mixture_parameters(self):
        """
            Generate a set of random mixture and shift parameters for each model that is in model_history.
        """ 
        mixture_weights = torch.rand(size=(self._num_transfer_functions, len(self.model_history)))
        mixture_weights = torch.softmax(mixture_weights, dim=-1)
        disc_shift_params, cont_shift_params = [], []
        curr_ssp_definition = self.search_space.get_search_space_definition()

        for _, domain, search_space, _, _ in self.model_history:
            ssp_definition = search_space.get_search_space_definition()
            domain = domain[:-1] # ignore score domain
            
            # init continuous shift parameters
            num_cont_params = len([d for d in domain if d.data_type == DataType.CONTINUOUS])
            cont_params = torch.zeros((self._num_transfer_functions, num_cont_params))
            curr_cont_hps = [k for k in curr_ssp_definition.keys() if curr_ssp_definition[k]['type'] == MetaType.REAL]
            cont_hps = [k for k in ssp_definition.keys() if ssp_definition[k]['type'] == MetaType.REAL]
            overlap_idx = [cont_hps.index(k) for k in curr_cont_hps if k in cont_hps]
            random_shifts = np.vstack([np.random.uniform(d.min, d.max, self._num_transfer_functions) for d in domain if d.data_type == DataType.CONTINUOUS]).T
            cont_params[:, overlap_idx] += torch.from_numpy(random_shifts)

            # init discrete shift parameters
            curr_disc_hps = [k for k in curr_ssp_definition.keys() if curr_ssp_definition[k]['type'] == MetaType.DISCRETE]
            disc_hps = [k for k in ssp_definition.keys() if ssp_definition[k]['type'] == MetaType.DISCRETE]
            disc_params = []

            for dhp in disc_hps:
                allowed_vals = ssp_definition[dhp]['allowed']
                allowed_shifts = (2*len(allowed_vals)) + 1
                if dhp in curr_disc_hps:
                    param = torch.from_numpy(np.random.uniform(0, 1, size=(self._num_transfer_functions, allowed_shifts)))
                else:
                    # case where the current search space does not contain dhp
                    param = torch.zeros(self._num_transfer_functions, allowed_shifts)
                    param[:, len(allowed_vals) + 1] = 1 # set the parameter corresponding to zero-shift to 1, rest remains zero
                disc_params.append(param)
            
            disc_shift_params.append(disc_params)
            cont_shift_params.append(cont_params.T)
        return disc_shift_params, cont_shift_params, mixture_weights
    
    def _log_r2(self):
        data = torch.clone(self.data)
        data[:, -1] = torch.nan
        gt = self.data[:,-1].numpy()
        if len(self.model_history) == 0:
            pred = self.einet.sample(evidence=data, is_mpe=True).squeeze()[:, -1].numpy()
            r2 = r2_score(gt.reshape(-1, 1), pred.reshape(-1, 1))
        else:
            regression_engine = MixtureEinetRegressionEngine(self.model_history, self.search_space, self.mixture_weights)
            scd_order_mixture_weights = torch.softmax(self.second_order_mixture_weights, dim=0)
            pred = regression_engine.predict(data, False)
            einet_pred = self.einet.sample(evidence=data).squeeze()[:, -1]
            pred = scd_order_mixture_weights[0] * pred + scd_order_mixture_weights[1] * einet_pred
            r2 = r2_score(gt.reshape(-1, 1), pred.detach().numpy().reshape(-1, 1))
        print(f"R2: {r2}")
        

    @property
    def history(self):
        return self._history
