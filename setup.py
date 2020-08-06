from setuptools import setup, find_packages
import sys
import subprocess

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

def remove_requirements(requirements, name, replace=''):
    new_requirements = []
    for requirement in requirements:
        if requirement.split(' ')[0] != name:
            new_requirements.append(requirement)
        elif replace is not None:
            new_requirements.append(replace)
    return new_requirements

if sys.platform in ['win32','cygwin','windows']:
    requirements = remove_requirements(requirements,'torch')

    print('Trying to install pytorch and torchvision!')
    code = 1
    try:
        code = subprocess.call(['pip', 'install', 'torch==1.6.0', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        if code != 0:
            raise Exception('Torch installation failed !')
    except:
        try:
            code = subprocess.call(['pip3', 'install', 'torch==1.6.0', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
            if code != 0:
                raise Exception('Torch installation failed !')
        except:
            print('Failed to install pytorch, please install pytorch and torchvision manually be following the simple instructions over at: https://pytorch.org/get-started/locally/')
    if code == 0:
        print('Successfully installed pytorch version!')


setup(name='breizhcrops',
      version='0.0.2.4',
      description='A Satellite Time Series Dataset for Crop Type Identification',
      url='http://github.com/dl4sits/breizhcrops',
      author='Marc Rußwurm, Charlotte Pelletier, Max Zollner',
      author_email='marc.russwurm@tum.de',
      license='MIT',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False)
