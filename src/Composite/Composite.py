"""Definition of meta model 'Composite'."""
from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'Composite'
nsURI = ''
nsPrefix = ''

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


@abstract
class Image(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

    def getText(self):

        raise NotImplementedError('operation getText(...) not yet implemented')


class ImagePart(Image):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def getText(self):

        raise NotImplementedError('operation getText(...) not yet implemented')
