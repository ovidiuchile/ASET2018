"""Definition of meta model 'ocr'."""
from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'ocr'
nsURI = 'http://www.example.org/ocr'
nsPrefix = 'ocr'

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)
