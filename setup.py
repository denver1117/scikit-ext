"""
Run setup
"""

from distutils.core import setup

setup(name='scikit-ext',
      version='0.1.1',
      description='Various scikit-learn extensions',
      url='https://github.com/denver1117/scikit-ext',
      download_url='https://github.com/denver1117/scikit-ext/archive/0.1.tar.gz',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=['scikit_ext'],
      install_requires=[
          'numpy>=1.13.1',
          'scipy>=0.17.0',
          'scikit-learn>=0.18.2'
      ])
