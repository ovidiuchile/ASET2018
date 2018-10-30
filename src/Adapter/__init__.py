from .Adapter import getEClassifier, eClassifiers
from .Adapter import name, nsURI, nsPrefix, eClass
from .Adapter import Engine, ImageBmp, ImageToBmp, Image


from . import Adapter
from .. import adapter


__all__ = ['Engine', 'ImageBmp', 'ImageToBmp', 'Image']

eSubpackages = []
eSuperPackage = adapter
Adapter.eSubpackages = eSubpackages
Adapter.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)
