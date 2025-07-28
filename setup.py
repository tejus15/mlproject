from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path: str)->List[str]:
    '''
        Returns list of all requirements
    '''
    requirements=[]
    # Read the requirements.txt file
    with open(file_path) as file_obj:
        # Read the contents of the requirements.txt file line by line
        requirements=file_obj.readlines()

        # This list contains requirements as well as newline character '\n'
        requirements=[req.replace('\n', '') for req in requirements]

        # Make sure to remove '-e .' from the list of requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    # Return list of requirements
    return requirements

setup(
    name='mlproject',
    version='1.0',
    author='Tejus',
    author_email='tejus98sharma@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
