from setuptools import setup
from distutils.util import convert_path

exec(open(convert_path('masktools/version.py')).read())

with open('README.md') as f:
    long_description = f.read()

setup(name='masktools',
      version=__version__,
      description='Tools for making DEIMOS slit masks',
      long_description=long_description,
      url='https://github.com/adwasser/masktools',
      download_url='https://github.com/adwasser/masktools/tarball/' + __version__,
      author='Asher Wasserman',
      author_email='adwasser@ucsc.edu',
      license='MIT',
      packages=['masktools', 'masktools/superskims'],
      package_data={'': ['LICENSE', 'README.md', 'masktools/version.py']},
      scripts=['bin/superskims'],
      include_package_data=True,
      install_requires=['numpy', 'matplotlib', 'astropy'],
      zip_safe=False)
