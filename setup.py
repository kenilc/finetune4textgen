from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'datasets>=1.8.0',
    'google-cloud-storage',
    'numpy>=1.18.5'
    'pyarrow>=3.0.0',
    'torch>=1.9.0',
    'tqdm',
    'sentencepiece>=0.1.95',
    'transformers>=4.8.2',
]

setup(
    name='trainer',
    version='0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True
)
