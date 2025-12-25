from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    
    HYPEN_E_DOT = '-e .'
    '''
    This function will return the list of requirements
    '''

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'MLproject',
    version = '0.0.1',
    description="this is my first end to end MLproject",
    author = 'Dash',
    author_email = 'tboy4198@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt") #This parameter you have to remember because there is no suggestion in this method
)