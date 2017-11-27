
test_priors: vae-ppca/priors.py
	python vae-ppca/priors.py -f

test_ppca: vae-ppca/ppca.py
	python vae-ppca/ppca.py -f

test: test_priors test_ppca

