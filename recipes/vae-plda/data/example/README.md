Data specific configuration of the recipe
=========================================

Training and testing data can be set by specifying a list of "npz"
archives in `train_archives` and `test_archives` respectively. Each
of the archives has to contain 2 keys: `features` and `labels` for the
features and the associated class category. The features should be
stored in a `NxD` matrix where `N` is the number of data points and
`D` is the dimension of the features. `labels` should be a list of
the class indices starting from `0`.

Parameters of the model can be set by editing `plda.yml` and
`vaeplda.yml`.
