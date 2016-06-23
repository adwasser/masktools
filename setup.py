from setuptools import setup

exec(open('version.py').read())

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
      package_data={'': ['LICENSE', 'README.md', 'version.py']},
      scripts=['bin/superskims'],
      include_package_data=True,
      install_requires=['numpy', 'astropy', 'astroquery'],
      zip_safe=False)
