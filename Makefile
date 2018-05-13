
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
	@rm -rf *.egg* build dist
	@find . -depth -type d -name "__pycache__" -exec rm -fr "{}" \;

doc:
	@$(MAKE) -C docs html

linting:
	@pylint --rcfile .pylintrc beer

test:
	@python tests/run_tests.py --nruns 10 --tensor-type float
	@python tests/run_tests.py --nruns 10 --tensor-type double


.PHONY: clean test
