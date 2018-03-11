
test_expfamily:
	python tests/test_expfamily.py -f -v

test_features:
	python tests/test_features.py -f -v

test_normal:
	python tests/test_normal.py -f -v

test_mixture:
	python tests/test_mixture.py -f -v


test_models: test_normal test_mixture
test: test_expfamily test_features test_models

