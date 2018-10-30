from .CompositeIteratorVisitor import getEClassifier, eClassifiers
from .CompositeIteratorVisitor import name, nsURI, nsPrefix, eClass
from .CompositeIteratorVisitor import Image, ImagePiece, ImageComposite, IteratorInterface, EngineInterface, ImageIterator, Model, NNModel


from . import CompositeIteratorVisitor
from .. import oCRmata


__all__ = ['Image', 'ImagePiece', 'ImageComposite', 'IteratorInterface',
           'EngineInterface', 'ImageIterator', 'Model', 'NNModel']

eSubpackages = []
eSuperPackage = oCRmata
CompositeIteratorVisitor.eSubpackages = eSubpackages
CompositeIteratorVisitor.eSuperPackage = eSuperPackage


otherClassifiers = []

for classif in otherClassifiers:
    eClassifiers[classif.name] = classif
    classif.ePackage = eClass

for classif in eClassifiers.values():
    eClass.eClassifiers.append(classif.eClass)

for subpack in eSubpackages:
    eClass.eSubpackages.append(subpack.eClass)