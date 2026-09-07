from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This fucntion will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='3.13.3',
    author='Arya Deep Chowdhury',
    author_email='arya.d.chowdhury@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)