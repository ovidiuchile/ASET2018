"""Definition of meta model 'Adapter'."""
from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'Adapter'
nsURI = ''
nsPrefix = ''

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


@abstract
class Engine(EObject, metaclass=MetaEClass):

    image = EReference(ordered=True, unique=True, containment=False)

    def __init__(self, *, image=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

        if image is not None:
            self.image = image


class ImageBmp(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

    def getTextFromBmp(self):

        raise NotImplementedError('operation getTextFromBmp(...) not yet implemented')


@abstract
class Image(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

    def getText(self):

        raise NotImplementedError('operation getText(...) not yet implemented')


class ImageToBmp(Image):

    image = EReference(ordered=True, unique=True, containment=False)

    def __init__(self, *, image=None, **kwargs):

        super().__init__(**kwargs)

        if image is not None:
            self.image = image

    def getText(self):

        raise NotImplementedError('operation getText(...) not yet implemented')

    def convertImage(self):

        raise NotImplementedError('operation convertImage(...) not yet implemented')
