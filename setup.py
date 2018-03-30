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
    ],
    version=1.0,
    packages=['beer', 'beer.models']
)


