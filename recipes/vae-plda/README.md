VAE-PLDA
=========

This recipe provide an example of use of the  Variational AutoEncode \- Probabilistic Discriminant Analysis (VAE-PLDA) model and a comparison with the standard PLDA model.

The directory is organized as follows:

```
./data
  ./database_name/
    # database specific files
./path.sh
./run.sh
./utils/
  # Scripts specific to the recipe
```

This recipe assume the Sun Grid Engine environment (`qsub` command) with a GPU available. To run the recipe without the GPU, comment all the occurences of `--use-gpu` in the header of the `./run.sh` file. If your system has a different scheduler than SGE, you will have to modify the `./run.sh` file and eventually the `./utils/job.qsub` file.

In order to run the recipe you will have to provide the data and to set the parameters of the model (see `./database_example/README.md`). Then, set the name of the database accordingly in `./run.sh` and, eventually, change the settings of recipe to fit your system resources. Finally, if you need to set a specific environment variables for the recipe define them in `./path.sh`.

Finally, when the settings is done, simply type:

```
$ ./run.sh
```

to run the recipe.
