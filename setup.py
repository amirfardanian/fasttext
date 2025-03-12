from setuptools import setup
import os

__local_dir = os.path.dirname(__file__)
version_filename = os.path.join(__local_dir, './capture_model/version.py')
exec (open(version_filename).read())  # loads __version__ variable

setup(
    name="capture_model",
    version=__version__,
    packages=['capture_model',
              'capture_model.modelling',
              'capture_model.validation',
              'capture_model.package_data'],
    install_requires=[
        'scipy==1.0.0',
        'scikit-learn==0.19.1',
        'numpy==1.13.3',
        'pandas==0.21.0',
        'cython==0.27.3',
        'cysignals==1.6.6',
        'fasttext==0.4.4',
    ],
    dependency_links=[],
    package_data={
        'capture_model': ['package_data/*']
    }
)
