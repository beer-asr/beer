Acoustic Unit Discovery
=======================

This recipe shows how to use *beer* to build an Acoustic Unit Discovery
system.

To run the recipe with the default settings type:

```
    ./run.sh
```

Basic configuration can be found in `conf/mfcc.yml`, `conf/hmm.yml`
and in the header of `run.sh`.

ZRC challenge
-------------

Under construction !!

Edit the `run.sh` file and set `db=zrc2019`. Also, specify the path
of the raw WAV files in `local/zrc2019/prepare_data.sh`.

Training
--------

By default the recipe will train the model on a single machine using stochastic training. To keep things fast, it uses Viterbi training instead of Baum-Welch training. If you environment has a SGE like cluster (i.e. `qsub` command) you can edit the `run.sh` file and uncomment the parallel training. 
