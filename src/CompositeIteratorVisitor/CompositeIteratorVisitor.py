"""Definition of meta model 'CompositeIteratorVisitor'."""
from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'CompositeIteratorVisitor'
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

    def accept(self):

        raise NotImplementedError('operation accept(...) not yet implemented')


@abstract
class IteratorInterface(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

    def first(self):

        raise NotImplementedError('operation first(...) not yet implemented')

    def next(self):

        raise NotImplementedError('operation next(...) not yet implemented')

    def isDone(self):

        raise NotImplementedError('operation isDone(...) not yet implemented')


@abstract
class EngineInterface(EObject, metaclass=MetaEClass):

    composite = EReference(ordered=True, unique=True, containment=False)
    iteratorinterface = EReference(ordered=True, unique=True, containment=False)

    def __init__(self, *, composite=None, iteratorinterface=None, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

        if composite is not None:
            self.composite = composite

        if iteratorinterface is not None:
            self.iteratorinterface = iteratorinterface


@abstract
class Model(EObject, metaclass=MetaEClass):

    def __init__(self, **kwargs):
        if kwargs:
            raise AttributeError('unexpected arguments: {}'.format(kwargs))

        super().__init__()

    def visitImagePiece(self):

        raise NotImplementedError('operation visitImagePiece(...) not yet implemented')

    def visitImageComposite(self):

        raise NotImplementedError('operation visitImageComposite(...) not yet implemented')


class ImagePiece(Image):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def getText(self):

        raise NotImplementedError('operation getText(...) not yet implemented')

    def accept(self):

        raise NotImplementedError('operation accept(...) not yet implemented')


class ImageComposite(Image):
    image = EReference(ordered=True, unique=True, containment=True, upper=-1)

    def __init__(self, *, image=None, **kwargs):

        super().__init__(**kwargs)

        if image:
            self.image.extend(image)

    def getText(self):

        raise NotImplementedError('operation getText(...) not yet implemented')

    def accept(self):

        raise NotImplementedError('operation accept(...) not yet implemented')


class ImageIterator(IteratorInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def first(self):
        raise NotImplementedError('operation first(...) not yet implemented')

    def next(self):
        raise NotImplementedError('operation next(...) not yet implemented')

    def isDone(self):
        raise NotImplementedError('operation isDone(...) not yet implemented')


class NNModel(Model):

    imagecomposite = EReference(ordered=True, unique=True, containment=False)
    imagepiece = EReference(ordered=True, unique=True, containment=False)

    def __init__(self, *, imagecomposite=None, imagepiece=None, **kwargs):

        super().__init__(**kwargs)

        if imagecomposite is not None:
            self.imagecomposite = imagecomposite

        if imagepiece is not None:
            self.imagepiece = imagepiece

    def visitImagePiece(self):
        """ Va returna litera identificata """
        raise NotImplementedError('operation visitImagePiece(...) not yet implemented')

    def visitImageComposite(self):
        """  Va parcurge imaginile cu cate o litera dintr-o imagine mai mare"""
        raise NotImplementedError('operation visitImageComposite(...) not yet implemented')
