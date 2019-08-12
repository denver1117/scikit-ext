"""
Run setup
"""

from distutils.core import setup
from os import path

try:
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.rst')) as f:
        long_description = f.read()
except:
    long_description = ""

setup(name='scikit-ext',
      version='0.1.10',
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
      download_url='https://github.com/denver1117/scikit-ext/archive/0.1.10.tar.gz',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=['scikit_ext'],
      install_requires=[
          'pandas',
          'numpy>=1.17.0',
          'scipy>=1.3.1',
          'scikit-learn>=0.21.3'
      ])
