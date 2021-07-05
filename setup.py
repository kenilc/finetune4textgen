from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'datasets',
    'pyarrow',
    'torch',
    'sentencepiece',
    'transformers',
]

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True
)
