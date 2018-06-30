
'''Run the all the tests.

This script should be run from the beer root directory.

'''

# pylint: disable=C0413
# Not all the modules can be placed at the top of the files as we need
# first to change the PYTHONPATH before to import the modules.
import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

import argparse
import unittest
from basetest import BaseTest
import test_create_model
import test_bayesmodel
import test_expfamilyprior
import test_features
import test_mixture
import test_normal
import test_hmm
import test_subspacemodels
import test_utils
import test_vbi

testcases = {
    'test_features': test_features,
    'test_expfamilyprior': test_expfamilyprior,
    'test_bayesmodel': test_bayesmodel,
    'test_create_model': test_create_model,
    'test_mixture': test_mixture,
    'test_normal': test_normal,
    'test_subspacemodels': test_subspacemodels,
    'test_utils': test_utils,
    'test_vbi': test_vbi,
    'test_hmm': test_hmm
}

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-seed', type=int, default=1,
                        help='initial seeding value for the random number '
                             'generator')
    parser.add_argument('--nruns', type=int, default=1,
                        help='number of runs (with different seed values) '
                             'for a test case')
    parser.add_argument('--tensor-type', choices=['float', 'double'],
                        default='float',
                        help='type of the tensor to use in the tests')
    parser.add_argument('--testcase', choices=list(testcases.keys()),
                        help='specific test case to run')
    parser.add_argument('--verbosity', choices=[1, 2], default=1, type=int,
                        help='verbosity')
    args = parser.parse_args()
    tensor_type = args.tensor_type
    init_seed = args.init_seed


    if args.testcase is not None:
        test_modules = [testcases[args.testcase]]
    else:
        test_modules = [
            test_bayesmodel,
            test_expfamilyprior,
            test_features,
            test_hmm,
            test_mixture,
            test_normal,
            test_subspacemodels,
            test_utils,
            test_vbi,

        ]

    suite = unittest.TestSuite()

    for test_module in test_modules:
        for testcase_name in test_module.__all__:
            testcase = getattr(test_module, testcase_name)
            for i in range(args.nruns):
                suite.addTest(BaseTest.get_testsuite(testcase,
                                                     tensor_type=tensor_type,
                                                     seed=init_seed + i))
    unittest.TextTestRunner(verbosity=args.verbosity, failfast=True).run(suite)


if __name__ == '__main__':
    run()
else:
    print('This script cannot be imported')
    exit(1)
