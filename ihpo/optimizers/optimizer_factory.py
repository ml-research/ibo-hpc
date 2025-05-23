from . import *

class OptimizerFactory:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OptimizerFactory, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_optimizer(cls, optimizer_name, optimizer_cfg):
        if optimizer_name == 'pc':
            return PCOptimizer(**optimizer_cfg)
        elif optimizer_name == 'pc_transfer':
            return PCTransOptimizer(**optimizer_cfg)
        elif optimizer_name == 'pc_transfer_rf':
            return PCTransRFOptimizer(**optimizer_cfg)
        elif optimizer_name == 'einet':
            return EinetOptimizer(**optimizer_cfg)
        elif optimizer_name == 'ls':
            return LocalSearchOptimizer(**optimizer_cfg)
        elif optimizer_name == 'rs':
            return RandomSearchOptimizer(**optimizer_cfg)
        elif optimizer_name == 'gp':
            return GPOptimizer(**optimizer_cfg)
        elif optimizer_name == 'smac':
            return SMACOptimizer(**optimizer_cfg)
        elif optimizer_name == '0shot':
            return ZeroShotOptimizer(**optimizer_cfg)
        elif optimizer_name == 'bbox':
            return BoundingBoxOptimizer(**optimizer_cfg)
        elif optimizer_name == 'pibo':
            return PiBOptimizer(**optimizer_cfg)
        elif optimizer_name == 'quant':
            return QuantileBasedOptimizer(**optimizer_cfg)
        elif optimizer_name == 'hyperband':
            return HyperbandOptimizer(**optimizer_cfg)
        elif optimizer_name == 'bopro':
            return BOPrOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'optunabo':
            return OptunaBOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'skoptbo':
            return SkOptBOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'rgpe' or optimizer_name == 'transbo':
            return OpenBoxOptimizer(**optimizer_cfg)
        elif optimizer_name == 'mphd':
            return MPHDOptimizer(**optimizer_cfg)
        elif optimizer_name == 'fsbo':
            return FSBOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'ablr':
            return ABLROptimizer(**optimizer_cfg)
        elif optimizer_name == 'het_mtgp':
            return HeterogeneousMTGP(**optimizer_cfg)
        else:
            raise ValueError(f'No such optimizer available: {optimizer_name}')