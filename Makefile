
PACKAGE_NAME = beer


default:
	@echo 'Makefile for $(PACKAGE_NAME)'
	@echo
	@echo 'Usage:'
	@echo '  make install    install the package in a new virtual environment'
	@echo '  make test       run the test suite'
	@echo '  make clean      clean up temporary files'


install:
	@python setup.py install


clean:
	@rm -rf *.egg build dist
	@find . -type d -name __pycache__ -delete


test:
	@python tests/test_expfamilyprior.py -f -v
	@python tests/test_bayesmodel.py -f -v
	@python tests/test_bayesembedding.py -f -v
	@python tests/test_features.py -f -v
	@python tests/test_normal.py -f -v
	@python tests/test_mixture.py -f -v
	@python tests/test_mlpmodel.py -f -v
	@python tests/test_vbi.py -f -v


.PHONY: clean test
