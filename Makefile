
test_expfamilyprior:
	python tests/test_expfamilyprior.py -f -v

test_features:
	python tests/test_features.py -f -v

test_normal:
	python tests/test_normal.py -f -v

test_mixture:
	python tests/test_mixture.py -f -v


test_models: test_normal test_mixture
test: test_expfamilyprior test_features test_models

