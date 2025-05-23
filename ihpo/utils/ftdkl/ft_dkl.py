import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ..gaussian_processes import SVGPModel
from ihpo.search_spaces import SearchSpace
from rtdl_revisiting_models import FTTransformer
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
import os
from ..acquisitions import EI

def totorch(x, device):
    return torch.Tensor(x, device=device)


class FTDeepKernelGP(nn.Module):
    def __init__(self, feature_extractor, inducing_points):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp = SVGPModel(inducing_points)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp(features)


class FTDKLModel:

    def __init__(self, data, hyperparameters, n_cont_features, cat_cardinalities, num_inducing, device, 
                 checkpoint_dir, model_name, eval_batch_size = 1000):
        self.data = data
        self.dataset = TensorDataset(data[:, :-1], data[:, -1])
        self.loader = DataLoader(self.dataset, batch_size=256)
        self.hyperparameters = hyperparameters
        self._gpu = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        self.eval_batch_size = eval_batch_size
        self._checkpoint_dir = checkpoint_dir
        self._model_file = os.path.join(checkpoint_dir, model_name + '.pt')
        if os.path.exists(self._model_file):
            self.dkgp = torch.load(self._model_file)
        else:
            self.ft_transformer = FTTransformer(
                    n_cont_features=n_cont_features,
                    cat_cardinalities=cat_cardinalities,
                    d_out=512,
                    n_blocks=3,
                    d_block=128,
                    attention_n_heads=8,
                    attention_dropout=0.2,
                    ffn_d_hidden=None,
                    ffn_d_hidden_multiplier=4 / 3,
                    ffn_dropout=0.1,
                    residual_dropout=0.0,
            ).to(self._gpu)
            self.ft_model = nn.Sequential(self.ft_transformer, nn.Linear(512, 1))
            inducing_points_idx = np.random.choice(np.arange(len(data)), num_inducing)
            inducing_points = torch.from_numpy(data[inducing_points_idx]).to(self._gpu)
            self.dkgp = FTDeepKernelGP(self.ft_transformer, inducing_points).to(self._gpu)
            self.mse_optim = torch.optim.adamw.AdamW(self.ft_transformer.parameters(), lr=1e-5)
            self.mse = nn.MSELoss()
        
        # initialize loss and optimizer for FT-Transformer with SVGP layer
        self.likelihood = GaussianLikelihood()
        self.mll = VariationalELBO(self.likelihood, self.dkgp.gp, num_data=num_inducing)

        self.optimizer = torch.optim.Adam(self.dkgp.parameters(), lr=0.01)


    def train(self):
        self.ft_model.train()
        self.dkgp.train()
        self.likelihood.train()

        for epoch in range(300):  # Number of epochs
            
            total_loss = 0.0
            for x, y in self.loader:
                
                x, y = x.to(self._gpu), y.to(self._gpu)
                # Step 1: Optimize the FT-Transformer with MSE loss
                self.mse_optim.zero_grad()
                features = self.ft_model(x)
                mse = self.mse(features, y.unsqueeze(-1))  # Reshape targets if needed
                mse.backward()
                self.mse_optim.step()

                total_loss += mse.item()

            print(f"Epoch: {epoch+1}/300 \t Loss (MSE): {total_loss / len(self.loader)}")

        for epoch in range(50):
            # Step 2: Optimize the GP with ELBO
            total_loss = 0.0
            for x, y in self.loader:
                
                x, y = x.to(self._gpu), y.to(self._gpu)
                self.optimizer.zero_grad()
                features = self.ft_transformer(x)
                gp_output = self.dkgp.gp(features)
                elbo_loss = -self.mll(gp_output, y)
                elbo_loss.backward()
                self.optimizer.step()

                total_loss += elbo_loss.item()
            
            print(f"Epoch: {epoch+1}/50 \t Loss (ELBO): {total_loss / len(self.loader)}")

        torch.save(self.dkgp, self._model_file)

    def _fine_tune(self, X_obs, y_obs):
        dataset = TensorDataset(X_obs, y_obs)
        loader = DataLoader(X_obs, y_obs, batch_size=256)
        for epoch in range(50):
            # Optimize the GP with ELBO using freshly observed data
            total_loss = 0.0

            for x, y in loader:
                x, y = x.to(self._gpu), y.to(self._gpu)
                self.optimizer.zero_grad()
                features = self.ft_transformer(x)
                gp_output = self.dkgp.gp(features)
                elbo_loss = -self.mll(gp_output, y)
                elbo_loss.backward()
                self.optimizer.step()

                total_loss += elbo_loss.item()
            
            print(f"Epoch: {epoch+1}/50 \t Loss (ELBO): {total_loss / len(self.loader)}")

    def predict(self, x):

        with torch.no_grad():
            features = self.ft_transformer(x)
            gp_output = self.dkgp.gp(features)
        return gp_output

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        X_obs, y_obs, X_pen = totorch(X_obs, self._gpu), totorch(y_obs, self._gpu).reshape(-1), totorch(X_pen, self._gpu)
        n_samples = len(X_pen)
        scores = []

        self._fine_tune(X_obs, y_obs)

        for i in range(self.eval_batch_size, n_samples+self.eval_batch_size, self.eval_batch_size):
            temp_X = X_pen[range(i-self.eval_batch_size,min(i,n_samples))]
            mu, stddev = self.predict(temp_X)
            score   =  EI(max(y_obs), mu, stddev)
            scores += score.tolist()

        scores = np.array(scores)
        candidate = np.argmax(scores) 

        return candidate

    def mix_up_initialize(self, search_space: SearchSpace):
        """
            If an unseen dimension is part of the search space, perform mix-up initialization of FT layer as
            described in https://openreview.net/pdf?id=0aAd19ZQp11.
        """
        ssd = search_space.get_search_space_definition()
        new_hps = [(name, val) for name, val in ssd.items() if name not in self.hyperparameters]

        if len(new_hps) == 0:
            return
        
        new_cat_hps = [(name, val) for (name, val) in new_hps if val['dtype'] != 'float']
        new_cont_hps = [(name, val) for (name, val) in new_hps if val['dtype'] == 'float']
        
        # ============== Initialize categorical embeddings ===============
        # Original embedding layer
        embedding_layer = self.ft_transformer.tokenizer.category_embeddings
        original_input_dim = embedding_layer.num_embeddings
        embedding_dim = embedding_layer.embedding_dim

        # Extend the input dimension
        new_input_dim = original_input_dim + len(new_cat_hps)
        extended_embedding = nn.Embedding(num_embeddings=new_input_dim, embedding_dim=embedding_dim)

        # Copy existing weights
        with torch.no_grad():
            extended_embedding.weight[:original_input_dim] = embedding_layer.weight  # Copy old weights
            # choose two random embedding vectors and a uniformly sampled weight for initialization

            for cat_hp_idx in range(0, len(new_cat_hps)):
                rnd_idx1, rnd_idx2 = np.random.randint(0, embedding_layer.weight.shape[0], size=2).tolist()
                rnd_weight = np.random.uniform(0, 1)[0]
                mixup_embedding_weights = (rnd_weight * embedding_layer.weight[rnd_idx1]) + ((1 - rnd_weight) * embedding_layer.weight[rnd_idx2])
                extended_embedding.weight[(original_input_dim + cat_hp_idx)] = mixup_embedding_weights

        self.ft_transformer.tokenizer.category_embeddings = extended_embedding

        # ============== Initialize continuous embeddings ===============
        # Original embedding layer
        embedding_weight = self.ft_transformer.tokenizer.weight
        original_input_dim = embedding_weight.shape[0]
        embedding_dim = embedding_weight.shape[1]

        # Extend the input dimension
        new_input_dim = original_input_dim + len(new_cont_hps)
        extended_embedding_weight = nn.Parameter(torch.randn((new_input_dim, embedding_dim)))

        # Copy existing weights
        with torch.no_grad():
            extended_embedding_weight[:original_input_dim] = embedding_weight
            # choose two random embedding vectors and a uniformly sampled weight for initialization

            for cont_hp_idx in range(0, len(new_cont_hps)):
                rnd_idx1, rnd_idx2 = np.random.randint(0, original_input_dim, size=2).tolist()
                rnd_weight = np.random.uniform(0, 1)[0]
                mixup_embedding_weights = (rnd_weight * embedding_weight[rnd_idx1]) + ((1 - rnd_weight) * embedding_weight[rnd_idx2])
                extended_embedding_weight[(original_input_dim + cont_hp_idx)] = mixup_embedding_weights

        self.ft_transformer.tokenizer.weight = extended_embedding_weight

        # =========== Adjust bias dimensions ============
        bias = self.ft_transformer.tokenizer.bias
        original_bias_in_dim = bias.shape[0]
        original_bias_out_dim = bias.shape[1]

        new_bias = nn.Parameter(torch.randn(new_input_dim, original_bias_out_dim))
        new_bias[:original_bias_in_dim] = bias
        self.ft_transformer.tokenizer.bias = new_bias