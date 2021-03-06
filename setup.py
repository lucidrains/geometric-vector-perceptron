from setuptools import setup, find_packages

setup(
  name = 'geometric-vector-perceptron',
  packages = find_packages(),
  version = '0.0.11',
  license='MIT',
  description = 'Geometric Vector Perceptron - Pytorch',
  author = 'Phil Wang, Eric Alcaide',
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
    'torch>=1.6',
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
    'torch-geometric'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
