from .search_space import SearchSpace
from naslib.search_spaces import NasBench101SearchSpace, \
    NasBench201SearchSpace, TransBench101SearchSpaceMicro
from typing import Union

class NASLibSearchSpace(SearchSpace):

    def __init__(self, benchmark: Union[NasBench201SearchSpace, 
                                        NasBench101SearchSpace, 
                                        TransBench101SearchSpaceMicro],
                        dataset_api) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.dataset_api = dataset_api

    def to_dict(self,  benchmark: Union[NasBench201SearchSpace, 
                                        NasBench101SearchSpace, 
                                        TransBench101SearchSpaceMicro]):
        raise NotImplementedError('Base class does not implement to_dict')
    
    def get_neighbors(self, architecture):
        if architecture is None:
            rand_archs = self.sample(size=5)
            return rand_archs
        spec = self._create_spec(architecture)
        new_bench = self.benchmark.clone()
        new_bench.set_spec(spec)
        nbhd = new_bench.get_nbhd(self.dataset_api)
        samples = [self.to_dict(n.arch) for n in nbhd]
        return samples
    
    def _create_spec(self, cfg):
        raise NotImplementedError('Base class does not implement _create_spec')
    
    @property
    def operations(self):
        raise NotImplementedError('Base class does not implement operations')
    
