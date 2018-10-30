
from .Composite import getEClassifier, eClassifiers
from .Composite import name, nsURI, nsPrefix, eClass
from .Composite import Image, ImagePart


from . import Composite
from .. import ocr


__all__ = ['Image', 'ImagePart']

eSubpackages = []
eSuperPackage = ocr
Composite.eSubpackages = eSubpackages
Composite.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
