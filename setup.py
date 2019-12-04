"""
Run setup
"""

from os import path
from setuptools import setup, find_packages
from scikit_ext import __version__

try:
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.rst')) as f:
        long_description = f.read()
except:
    long_description = ""

setup(name='scikit-ext',
      version=__version__,
      description='Various scikit-learn extensions',
      long_description=long_description,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      url='https://github.com/denver1117/scikit-ext',
      download_url='https://pypi.org/project/scikit-ext/#files',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.5",
      install_requires=[
          'pandas',
          'numpy>=1.17.0',
          'scipy>=1.3.1',
          'scikit-learn>=0.22.0'
      ])
