from setuptools import setup, find_packages

try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]

setup(name='breizhcrops',
      version='0.01',
      description='A Satellite Time Series Dataset for Crop Type Identification',
      url='http://github.com/dl4sits/breizhcrops',
      author='Marc Rußwurm, Charlotte Pelletier',
      author_email='marc.russwurm@tum.de',
      license='MIT',
      packages=find_packages(),
      install_requires=load_requirements("requirements.txt"),
      zip_safe=False)
