'''Run the all the tests. 

This script should be run from the beer root directory.

'''

import argparse
import unittest

import sys
sys.path.insert(0, './')
sys.path.insert(0, './tests')

from basetest import BaseTest
import test_expfamilyprior


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-seed', type=int, default=1, 
        help='initial seeding value for the random number generator')
    parser.add_argument('--nruns', type=int, default=1, 
        help='number of runs (with different seed values) for a test case')
    parser.add_argument('--tensor-type', choices=['float', 'double'], 
        default='float', help='type of the tensor to use in the tests')
    args = parser.parse_args()

    test_modules = [test_expfamilyprior]
    for test_module in test_modules:
        for testcase in test_module.__all__:
            suite = unittest.TestSuite()
            for i in range(args.nruns):
                suite.addTest(BaseTest.get_testsuite(testcase, 
                    tensor_type=args.tensor_type, seed=args.init_seed + i))
            unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)


if __name__ == '__main__':
    run()
else:
    print('This script cannot be imported')
    exit(1)
