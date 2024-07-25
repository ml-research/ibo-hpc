class SearchSpace:

    """
        Represents a search space of an HPO problem and provides a unified
        API for certain operations on the search space such as sampling and
        getting neigbor configurations.
        Usually used to sample random configurations at the beginning of
        an optimization run.
    """

    def __init__(self) -> None:
        pass

    def sample(self, **kwargs):
        raise NotImplementedError('Base class does not implement sample')
    
    def get_neighbors(self, config):
        raise NotImplementedError('Base class does not implement get_neighbours')
    
    def get_search_space_definition(self):
        raise NotImplementedError('Base class does not implement get_feature_space_definition')

    def to_configspace(self):
        raise NotImplementedError('Base class does not implement to_configspace')
    
    def to_synetune(self):
        raise NotImplementedError('Base class does not implement to_synetune')
    
    def to_hypermapper(self):
        raise NotImplementedError('Base class does not implement to_hypermapper')
    
    def is_valid(self, config):
        raise NotImplementedError('Base class does not implement is_valid')