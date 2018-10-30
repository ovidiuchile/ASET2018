from .Memento import getEClassifier, eClassifiers
from .Memento import name, nsURI, nsPrefix, eClass
from .Memento import Model, NNModel, NNEngine, Engine


from . import Memento
from .. import memento


__all__ = ['Model', 'NNModel', 'NNEngine', 'Engine']

eSubpackages = []
eSuperPackage = memento
Memento.eSubpackages = eSubpackages
Memento.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
