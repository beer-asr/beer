
test_expfamily:
	python tests/test_expfamily.py

test_features:
	python tests/test_features.py

test_normal:
	python tests/test_normal.py

test_mixture:
	python tests/test_mixture.py


test_models: test_normal test_mixture
test: test_expfamily test_features test_models

