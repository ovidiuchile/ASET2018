"""Definition of meta model 'Memento'."""
from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'Memento'
nsURI = ''
nsPrefix = ''

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


@abstract
class Model(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()


@abstract
class Engine(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()


class NNModel(Model):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def getState(self):

        raise NotImplementedError('operation getState(...) not yet implemented')

    def setState(self):

        raise NotImplementedError('operation setState(...) not yet implemented')


class NNEngine(Engine):

    nnmodel = EReference(ordered=True, unique=True, containment=False)

    def __init__(self, *, nnmodel=None, **kwargs):

        super().__init__(**kwargs)

        if nnmodel is not None:
            self.nnmodel = nnmodel

    def createMemento(self):

        raise NotImplementedError('operation createMemento(...) not yet implemented')

    def restore(self):

        raise NotImplementedError('operation restore(...) not yet implemented')
