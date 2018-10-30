
from .ocr import getEClassifier, eClassifiers
from .ocr import name, nsURI, nsPrefix, eClass


from . import ocr
from . import Strategy

from . import Singleton

from . import Composite


__all__ = []

eSubpackages = [Strategy, Singleton, Composite]
eSuperPackage = None
ocr.eSubpackages = eSubpackages
ocr.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
