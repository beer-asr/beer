Acoustic Unit Discovery
=======================

This recipe shows how to use **Beer** to build an Acoustic Unit Discovery
system.


Directory Structure
-------------------

* `conf`:  Directory containing the configuration files for the AUD
model and the features.
* `local`: Directory containing the data preparation scripts for various
corpora.
* `steps`: Directory containing scripts dedicated to one step of the
AUD pipe (features extraction, decoding, ...)
* `utils`: Directory contain miscellaneous scripts.


Combining Corpus
----------------

It is possible to combine several corpora together. This is useful, for
instance, when one wants to train a multi-lingual phone recognizer
system which can be used to build stronger AUD system. Let's assume
we have a corpora composed of an german and french parts stored in
`data/corpora/GE` and `data/corpora/FR` respectively. We can create
a new corpora combining both of them in the following way:

```bash
mkdir data/corpora/GE_FR
utils/add_corpus.sh --max-utts 1000 data/corpora/GE data/corpora/GE_FR
utils/add_corpus.sh --max-utts 1000 data/corpora/FR data/corpora/GE_FR
```

The option `--max-utts` specify the maximum number of utterances to
keep from the corpus. Note that selection of the utterances will be
random.

