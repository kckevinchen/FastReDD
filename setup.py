"""
Setup script for ReDD (Relational Deep Dive) project.
This setup.py allows the project to be installed as a package,
which resolves the ModuleNotFoundError in WSL and other environments.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    requirements.append(line)
    
    return requirements

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name='redd',
    version='1.0.1',
    description='ReDD: Error-Aware Queries Over Unstructured Data',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='ReDD Team',
    url='https://github.com/your-org/ReDD_Dev',
    
    # Find all packages in the project
    packages=find_packages(exclude=[
        'tests',
        'tests.*',
        'archive',
        'archive.*',
        'compared_algs',
        'compared_algs.*',
        'experiments',
        'experiments.*',
        'notebooks',
        'notebooks.*',
        'outputs',
        'outputs.*',
        'logs',
        'logs.*',
        'fig',
        'fig.*',
        'papers',
        'papers.*',
    ]),
    
    # Include package data (if needed)
    include_package_data=True,
    
    # Install dependencies from requirements.txt
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires='>=3.10',
    
    # Entry points for command-line scripts (optional)
    # Uncomment if you want to create command-line tools:
    # entry_points={
    #     'console_scripts': [
    #         'redd-datapop=scripts.main_datapop:main',
    #         'redd-schemagen=scripts.main_schemagen:main',
    #     ],
    # },
    
    # Classifiers for PyPI (if publishing)
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)

