"""Definition of meta model 'Strategy'."""
from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'Strategy'
nsURI = ''
nsPrefix = ''

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


@abstract
class Strategy(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

    def execute(self):

        raise NotImplementedError('operation execute(...) not yet implemented')


class NeuralNetworkStrategy(Strategy):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def execute(self):

        raise NotImplementedError('operation execute(...) not yet implemented')


class OCRadStrategy(Strategy):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def execute(self):

        raise NotImplementedError('operation execute(...) not yet implemented')
