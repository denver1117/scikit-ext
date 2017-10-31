"""
Run setup
"""

from distutils.core import setup
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

setup(name='scikit-ext',
      version='0.1.7',
      description='Various scikit-learn extensions',
      long_description=long_description,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      url='https://github.com/denver1117/scikit-ext',
      download_url='https://github.com/denver1117/scikit-ext/archive/0.1.7.tar.gz',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=['scikit_ext'],
      install_requires=[
          'pandas',
          'numpy>=1.13.1',
          'scipy>=0.17.0',
          'scikit-learn>=0.18.2'
      ])
