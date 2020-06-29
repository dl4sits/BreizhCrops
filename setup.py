from setuptools import setup, find_packages

setup(name='breizhcrops',
      version='0.0.2.4',
      description='A Satellite Time Series Dataset for Crop Type Identification',
      url='http://github.com/dl4sits/breizhcrops',
      author='Marc Rußwurm, Charlotte Pelletier, Max Zollner',
      author_email='marc.russwurm@tum.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            "geopandas>=0.5.0",
            "numpy>=1.17.0",
            "pandas>=0.24.2",
            "geojson>=2.4.1",
            "jupyter>=1.0.0",
            "matplotlib>=3.1.0",
            "seaborn>=0.9.0",
            "torch>=1.4.0",
            "tqdm>=4.32.2",
            "scikit-learn",
            "h5py",
            "requests"
      ],
      zip_safe=False)
