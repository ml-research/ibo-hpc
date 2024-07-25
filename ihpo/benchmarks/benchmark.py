from typing import Dict

class BenchQueryResult(dict):

    SUPPORTED_METRICS = ['train_performance', 'val_performance', 'test_performance', 
                'cost', 'train_loss', 'val_loss', 'test_loss', 'epochs']

    def __init__(self, train_performance, val_performance, 
                 test_performance, train_loss=None, val_loss=None, 
                 test_loss=None, cost=None, epochs=None) -> None:
        self._train_perf = train_performance
        self._val_perf = val_performance
        self._test_pref = test_performance
        self._train_loss = train_loss
        self._val_loss = val_loss
        self._test_loss = test_loss
        self._cost = cost
        self._epochs = epochs

    def _to_string(self):
        return "Benchmark Query Result:\n" +\
            f"Train-Performance: {self._train_perf}\n" +\
            f"Val-Performance: {self._val_perf}\n" +\
            f"Test-Performance: {self._test_pref}\n" +\
            f"Cost: {self._cost}"
    
    def __repr__(self) -> str:
        return self._to_string()

    def __str__(self) -> str:
        return self._to_string()
    
    def __getitem__(self, key):
        if key == 'train_performance':
            return self.train_performance
        elif key == 'val_performance':
            return self.val_performance
        elif key == 'test_performance':
            return self.test_performance
        elif key == 'train_loss':
            return self.train_loss
        elif key == 'val_loss':
            return self.val_loss
        elif key == 'test_loss':
            return self.test_loss
        elif key == 'cost':
            return self.cost
        elif key == 'epochs':
            return self.epochs
        else:
            raise KeyError(f'No such key: {key}')

    def set_cost(self, cost):
        """
            Sometimes cost are experiment-dependent (e.g. we want cummulative costs of multiple runs in LS instead of cost of a single configuration).
            This allows us to be flexible and allows to set the costs as needed.
        """
        self._cost = cost

    @property
    def train_performance(self):
        return self._train_perf
    
    @property
    def val_performance(self):
        return self._val_perf
    
    @property
    def test_performance(self):
        return self._test_pref
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def train_loss(self):
        return self._train_loss

    @property
    def val_loss(self):
        return self._val_loss
    
    @property
    def test_loss(self):
        return self._test_loss
    
    @property
    def epochs(self):
        return self._epochs


class BaseBenchmark:

    def __init__(self) -> None:
        pass

    def query(self, cfg: Dict) -> BenchQueryResult:
        raise NotImplementedError('BaseBenchmark has no implementation for query')
    
    @property
    def search_space(self):
        raise NotImplementedError('BaseBenchmark has no search space')