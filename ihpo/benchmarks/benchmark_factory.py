from . import *

class BenchmarkFactory:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BenchmarkFactory, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_benchmark(cls, name, benchmark_cfg):
        if name == 'hpo':
            return HPOTabularBenchmark(**benchmark_cfg)
        elif name == 'nas101':
            return NAS101Benchmark(**benchmark_cfg)
        elif name == 'nas201':
            return NAS201Benchmark(**benchmark_cfg)
        elif name == 'jahs':
            return JAHSBenchmark(**benchmark_cfg)
        elif name == 'transnas':
            return TransNASBench(**benchmark_cfg)
        elif name == 'hpob':
            return HPOBBenchmark(**benchmark_cfg)
        elif name == 'fcnet':
            return FCNetBenchmark(**benchmark_cfg)
        elif name == 'pd1':
            return PD1Benchmark(**benchmark_cfg)
        elif name == 'lcbench':
            return LCBenchmark(**benchmark_cfg)
        else:
            raise ValueError(f'Benchmark {name} not known.')