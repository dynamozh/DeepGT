from distutils.core import setup
from setuptools import find_packages
import os


setup(
	# Name of the package
	name='graphgps',
	# Packages to include into the distribution
	packages=find_packages('.'),
	# Start with a small number and increase it with
	# every change you make https://semver.org
	version='1.2.0',
	# Chose a license from here: https: //
	# help.github.com / articles / licensing - a -
	# repository. For example: MIT
	license='',
	# Short description of your library
	description='',
	# Long description of your library
	long_description='',
	# Your name
	author='',
	# Your email
	author_email='',
	# Either the link to your github or to your website
	url='',
	# Link from which the project can be downloaded
	download_url='',
	# List of keywords
	keywords=[],
	# List of packages to install with this one
	install_requires=[],
	# https://pypi.org/classifiers/
	classifiers=[]
)
