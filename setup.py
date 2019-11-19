from setuptools import setup

# For any additional packages, please add them here.
REQS = [
    "futures==3.1.1",
    "yapf==0.24.0",
]

setup(
    name='moneycars',
    version='0.0.1',
    packages=[
        'moneycars'
    ],
    url='https://github.com/dmadisetti/CS682',
    tests_require=['nose'],
    license='No License for now. Please contact authors.',
    author='Tianyu Ding, Ivan Liao, Dylan Madisetti, Alex Sun',
    author_email='madisetti@jhu.edu',
    description=("This project seeks to use SOTA unsupervised techniques to"
    "develop a competive pipeline for the PKU kaggle competition."),
    install_requires=REQS,
)
