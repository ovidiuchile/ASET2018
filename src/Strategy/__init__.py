
from .Strategy import getEClassifier, eClassifiers
from .Strategy import name, nsURI, nsPrefix, eClass
from .Strategy import NeuralNetworkStrategy, Strategy, OCRadStrategy


from . import Strategy
from .. import ocr


__all__ = ['NeuralNetworkStrategy', 'Strategy', 'OCRadStrategy']

eSubpackages = []
eSuperPackage = ocr
Strategy.eSubpackages = eSubpackages
Strategy.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
