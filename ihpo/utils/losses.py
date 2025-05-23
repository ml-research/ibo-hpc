import itertools
import torch

class StatefulDifferentiableRankingLoss:
    """
        Stateful differentiable ranking loss.
        NOTE: Calling this loss introduces a state that fixes true_y to boost performance. 
            If true_y changes during iterations, do not use this loss. Instead, use DifferentiableRankingLoss!
    """
    def __init__(self):
        self.pairs = None

    def __call__(self, true_y, pred_y):
        if self.pairs is None:
            # Compute the rank loss for varied loss function.
            comb = itertools.combinations(range(true_y.shape[0]), 2)
            self.pairs = list()
            # Compute the pairs.
            for _, (i, j) in enumerate(comb):
                if true_y[i] > true_y[j]:
                    self.pairs.append((i, j))
                elif true_y[i] < true_y[j]:
                    self.pairs.append((j, i))
        return self._evaluate(true_y, pred_y)

    def _evaluate(self, true_y, pred_y):
        loss = 0.
        pair_num = len(self.pairs)
        if pair_num == 0:
            return torch.tensor(0.)
        for (i, j) in self.pairs:
            loss += torch.log(1 + torch.exp(pred_y[j] - pred_y[i]))
        return loss/pair_num
    

class DifferentiableRankingLoss:
    """
        Stateless differentiable ranking loss.
    """

    def __call__(self, true_y, pred_y):
        # Compute the rank loss for varied loss function.
        comb = itertools.combinations(range(true_y.shape[0]), 2)
        self.pairs = list()
        # Compute the pairs.
        for _, (i, j) in enumerate(comb):
            if true_y[i] > true_y[j]:
                self.pairs.append((i, j))
            elif true_y[i] < true_y[j]:
                self.pairs.append((j, i))
        return self._evaluate(true_y, pred_y)

    def _evaluate(self, true_y, pred_y):
        loss = 0.
        pair_num = len(self.pairs)
        if pair_num == 0:
            return 0.
        for (i, j) in self.pairs:
            loss += torch.log(1 + torch.exp(pred_y[j] - pred_y[i]))
        return loss/pair_num