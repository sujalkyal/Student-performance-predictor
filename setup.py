from setuptools import find_packages,setup
from typing import List

MINUS_E_DOT='-e .'
def get_requirments(file_path:str)->List[str]:
    '''
    this function will return the list of requirements.txt
    '''
    require=[]
    with open(file_path) as file_obj:
        require=file_obj.readlines()
        require=[req.replace('\n','') for req in require]
        if MINUS_E_DOT in require:
            require.remove(MINUS_E_DOT)
    return require


setup(
    name='ml - projects',
    version='0.1.1',
    author='sujal kyal',
    author_email='sujalkyal2704@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirements.txt')
)