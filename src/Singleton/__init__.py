
from .Singleton import getEClassifier, eClassifiers
from .Singleton import name, nsURI, nsPrefix, eClass
from .Singleton import DatabaseConnection


from . import Singleton
from .. import ocr


__all__ = ['DatabaseConnection']

eSubpackages = []
eSuperPackage = ocr
Singleton.eSubpackages = eSubpackages
Singleton.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
