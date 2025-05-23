import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import copy 
import numpy as np
import os
import time
import gpytorch
from ..acquisitions import EI
from ..gaussian_processes import ExactGPModel

def totorch(x, device):
    return torch.Tensor(x, device=device)


class DeepKernelGP(nn.Module):
    def __init__(self, input_size, seed, hidden_size = [32,32,32,32],
                         max_patience = 16, kernel="matern", ard = False, nu =2.5, loss_tol = 0.0001,
                         lr = 0.001, load_model = False, checkpoint = None, epochs = 10000,
                         verbose = False, eval_batch_size = 1000):
        super(DeepKernelGP, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_extractor = MLP(self.input_size, hidden_size = self.hidden_size).to(self.device)
        self.kernel_config = {"kernel": kernel, "ard": ard, "nu": nu}
        self.max_patience = max_patience
        self.lr  = lr
        self.load_model = load_model
        assert checkpoint != None, "Provide a checkpoint"
        self.checkpoint = checkpoint
        self.epochs = epochs
        self.verbose = verbose
        self.loss_tol = loss_tol
        self.eval_batch_size = eval_batch_size

        self.get_model_likelihood_mll(1)

    def get_model_likelihood_mll(self, train_size):
        
        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x=train_x, train_y=train_y, likelihood=likelihood)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)



    def train(self):

        if self.load_model:
            assert(self.checkpoint is not None)
            print("Model_loaded")
            self.load_checkpoint(os.path.join(self.checkpoint,"weights"))
            

        losses = [np.inf]
        best_loss = np.inf
        starttime = time.time()
        weights = copy.deepcopy(self.state_dict())
        patience=0
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.lr},
                                {'params': self.feature_extractor.parameters(), 'lr': self.lr}])
                    
        for _ in range(self.epochs):
            optimizer.zero_grad()
            z = self.feature_extractor(self.X_obs)
            self.model.set_train_data(inputs=z, targets=self.y_obs, strict=False)
            predictions = self.model(z)
            try:
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Exception {e}")
                break
            
            if self.verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_+1,epochs=self.epochs,loss=loss.item(),noise=self.likelihood.noise.item()))                
            losses.append(loss.detach().to("cpu").item())
            if best_loss>losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())
            if np.allclose(losses[-1],losses[-2],atol=self.loss_tol):
                patience+=1
            else:
                patience=0
            if patience>self.max_patience:
                break
        self.load_state_dict(weights)
        print(f"Current Iteration: {len(self.y_obs)} | Incumbent {max(self.y_obs)} | Duration {np.round(time.time()-starttime)} | Epochs {_} | Noise {self.likelihood.noise.item()}")
        return losses
    
    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint,map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'],strict=False)
        self.likelihood.load_state_dict(ckpt['likelihood'],strict=False)
        self.feature_extractor.load_state_dict(ckpt['net'],strict=False)
        

    def predict(self, X_pen):
        
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()        

        z_support = self.feature_extractor(self.X_obs).detach()
        self.model.set_train_data(inputs=z_support, targets=self.y_obs, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(X_pen).detach()
            pred    = self.likelihood(self.model(z_query))

            
        mu    = pred.mean.detach().to("cpu").numpy().reshape(-1,)
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1,)
        
        return mu,stddev

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        self.X_obs, self.y_obs, X_pen = totorch(X_obs, self.device), totorch(y_obs, self.device).reshape(-1), totorch(X_pen, self.device)
        n_samples = len(X_pen)
        scores = []

        self.train()

        for i in range(self.eval_batch_size, n_samples+self.eval_batch_size, self.eval_batch_size):
            temp_X = X_pen[range(i-self.eval_batch_size,min(i,n_samples))]
            mu, stddev = self.predict(temp_X)
            score   =  EI(max(y_obs), mu, stddev)
            scores += score.tolist()

        scores = np.array(scores)
        candidate = np.argmax(scores) 

        return candidate

    
class FSBO(nn.Module):
    def __init__(self, train_data,valid_data, checkpoint_path, batch_size = 64, test_batch_size = 64,
                 n_inner_steps = 1, kernel = "matern", ard = False, nu=2.5, hidden_size = [32,32,32,32], seed=0):
        super(FSBO, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.RandomQueryGenerator= np.random.RandomState(seed)
        self.RandomSupportGenerator= np.random.RandomState(seed)
        self.RandomTaskGenerator = np.random.RandomState(seed)
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.n_inner_steps = n_inner_steps
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        first_dataset = list(self.train_data.keys())[0]
        self.input_size = len(train_data[first_dataset]["X"][0])
        self.hidden_size = hidden_size
        self.feature_extractor =  MLP(self.input_size, hidden_size = self.hidden_size).to(self.device)
        self.kernel_config = {"kernel": kernel, "ard": ard, "nu": nu}
        self.get_model_likelihood_mll(self.batch_size)
        self.mse = nn.MSELoss()
        self.curr_valid_loss = np.inf
        self.get_tasks()     
        
    def get_tasks(self,):

        self.tasks = list(self.train_data.keys())
        self.valid_tasks = list(self.valid_data.keys())
        

    def get_model_likelihood_mll(self, train_size):
        
        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x = train_x, train_y = train_y, likelihood = likelihood)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)


    def epoch_end(self):
        self.RandomTaskGenerator.shuffle(self.tasks)


    def meta_train(self, epochs = 50000, lr = 0.0001):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7)

        for epoch in range(epochs):
            print(f"[FSBO] Meta-train the Deep Kernel GP: Iter {epoch}/{epochs}")
            loss = self.train_loop(epoch, optimizer, scheduler)
            print(f"[FSBO] Loss: {loss}")
        
    def train_loop(self, epoch, optimizer, scheduler=None):
        
        self.epoch_end()
        assert(self.training)
        meta_mse = 0.0
        for task in self.tasks:
            inputs, labels = self.get_batch(task)
            task_mse = 0.0
            for _ in range(self.n_inner_steps):
                optimizer.zero_grad()
                z = self.feature_extractor(inputs)
                self.model.set_train_data(inputs=z, targets=labels, strict=False)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
                mse = self.mse(predictions.mean, labels)
                task_mse += mse
            meta_mse += task_mse / self.n_inner_steps
        meta_mse = meta_mse / len(self.tasks)
        if scheduler:
            scheduler.step()

        for task in self.valid_tasks:
            mse,loss = self.test_loop(task,train=False)
        self.feature_extractor.train()
        self.likelihood.train()
        self.model.train()
        return meta_mse
            
    def test_loop(self, task, train): # no optimizer needed for GP
        (x_support, y_support),(x_query,y_query) = self.get_support_and_queries(task,train)
        z_support = self.feature_extractor(x_support).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()        
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            loss = -self.mll(pred, y_query)
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query)

        return mse,loss

    def get_batch(self,task):
        # we want to fit the gp given context info to new observations
        # task is an algorithm/dataset pair
        Lambda,response =     np.array(self.train_data[task]["X"]), MinMaxScaler().fit_transform(np.array(self.train_data[task]["y"])).reshape(-1,)

        card, dim = Lambda.shape
        
        support_ids = self.RandomSupportGenerator.choice(np.arange(card),
                                              replace=False,size= min(self.batch_size, card))

        
        inputs,labels = Lambda[support_ids], response[support_ids]
        inputs,labels = totorch(inputs,device=self.device), totorch(labels.reshape(-1,),device=self.device)
        return inputs, labels
        
    def get_support_and_queries(self,task, train=False):
        
        # task is an algorithm/dataset pair
        
        hpo_data = self.valid_data if not train else self.train_data
        Lambda,response =     np.array(hpo_data[task]["X"]), MinMaxScaler().fit_transform(np.array(hpo_data[task]["y"])).reshape(-1,)
        card, dim = Lambda.shape

        support_ids = self.RandomSupportGenerator.choice(np.arange(card),
                                              replace=False,size=min(self.batch_size, card))
        diff_set = np.setdiff1d(np.arange(card),support_ids)
        query_ids = self.RandomQueryGenerator.choice(diff_set,replace=False,size=min(self.batch_size, len(diff_set)))
        
        support_x,support_y = Lambda[support_ids], response[support_ids]
        query_x,query_y = Lambda[query_ids], response[query_ids]
        
        return (totorch(support_x,self.device),totorch(support_y.reshape(-1,),self.device)),\
    (totorch(query_x,self.device),totorch(query_y.reshape(-1,),self.device))
        
    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=[32,32,32,32], dropout=0.0):
        
        super(MLP, self).__init__()
        self.nonlinearity = nn.ReLU()
        self.fc = nn.ModuleList([nn.Linear(in_features=input_size, out_features=hidden_size[0])])
        for d_out in hidden_size[1:]:
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))
        self.out_features = hidden_size[-1]
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        x = self.dropout(x)
        return x