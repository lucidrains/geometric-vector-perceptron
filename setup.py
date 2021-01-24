from setuptools import setup, find_packages

setup(
  name = 'geometric-vector-perceptron',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'Geometric Vector Perceptron - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/geometric-vector-perceptron',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'proteins',
    'biomolecules',
    'equivariance'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
