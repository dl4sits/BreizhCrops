from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setup(name='breizhcrops',
      version='0.0.2.8',
      description='A Satellite Time Series Dataset for Crop Type Identification',
      url='http://github.com/dl4sits/breizhcrops',
      author='Marc Rußwurm, Charlotte Pelletier, Max Zollner',
      author_email='marc.russwurm@tum.de',
      license='MIT',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False)
