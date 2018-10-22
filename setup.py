from setuptools import setup, find_packages

setup(
    author='Lucas Ondel, Karel Benes',
    description='Bayesian spEEch Recognizer',
    url='https://github.com/beer-asr/beer',
    name='beer',
    author_email='lucas.ondel@gmail.com',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'torch',
    ],
    version=1.0,
    packages=[
        'beer',
        'beer.priors',
        'beer.models',
        'beer.nnet',
        'beer.inference',
        'beer.cli',
        'beer.cli.subcommands',
        'beer.cli.subcommands.features',
        'beer.cli.subcommands.dataset',
    ],
    scripts=['beer/cli/beer']
)


