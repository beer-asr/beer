
test_models:
	python -m doctest docs/models.rst -f

test_priors:
	python -m doctest docs/priors.rst -f

test_vae:
	python -m doctest docs/vae.rst -f

test: test_priors test_models

