from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .' 

def get_requirements(file_path:str)->List[str]:
    """Reads a requirements file at the given file
    path and returns its contents as a list of strings.
    Args:
        file_path (str): The path to the requirements 
        file to be read.
    Returns:
        List[str]: A list of strings, each string representing a
        line of the requirements file.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Housing_price_prediction',
    version='0.0.1',
    author='Swinal',
    author_email='swinalwaghmare2802@gmail.com',
    install_requires = get_requirements(file_path='requirements.txt'),
    packages=find_packages()
)