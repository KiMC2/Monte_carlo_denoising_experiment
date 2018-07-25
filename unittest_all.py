import os
import unittest
from glob import glob

dir_name = 'tests'
testmodules = glob(os.path.join('./' + dir_name, "*_test.py"))

suite = unittest.TestSuite()

for t in testmodules:
    t = dir_name + '.' + os.path.basename(t).split('.')[0]
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

unittest.TextTestRunner().run(suite)