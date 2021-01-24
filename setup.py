from setuptools import setup, find_packages

setup(
  name = 'geometric-vector-perceptron',
  packages = find_packages(),
<<<<<<< HEAD
  version = '0.0.3',
=======
  version = '0.0.5',
>>>>>>> 21fcd8207e99fb79e4ee21e88bd57b7a136eddda
  license='MIT',
  description = 'Geometric Vector Perceptron - Pytorch',
  author = 'Phil Wang and Eric Alcaide',
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
    'torch_geometric'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
