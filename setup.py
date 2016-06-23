from setuptools import setup

exec(open('version.py').read())

with open('README.md') as f:
    long_description = f.read()

setup(name='masktools',
      version=__version__,
      description='Tools for making DEIMOS slit masks',
      long_description=long_description,
      url='http://github.com/adwasser/masktools',
      author='Asher Wasserman',
      author_email='adwasser@ucsc.edu',
      license='MIT',
      packages=['masktools'],
      package_data={'': ['LICENSE', 'README.md']},
      scripts=['bin/superskims'],
      include_package_data=True,
      install_requires=['numpy', 'astropy'],
      zip_safe=False)
