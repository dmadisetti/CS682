from setuptools import setup

# For any additional packages, please add them here.
REQS = [
    "futures==3.1.1",
    "yapf==0.24.0",
    "partd==1.1.0",
    "fsspec>=0.3.3",
    "torch>=1.3.1",
    "torchvision>=0.4.2",
    "dask==2.9.0",
    "Pillow>=6.2.1",
]

setup(
    name='carsmoney',
    version='0.0.1',
    packages=[
        'carsmoney'
    ],
    url='https://github.com/dmadisetti/CS682',
    tests_require=['nose'],
    include_package_data=True,
    license='No License for now. Please contact authors.',
    author='Tianyu Ding, Ivan Liao, Dylan Madisetti, Alex Sun',
    author_email='madisetti@jhu.edu',
    description=("This project seeks to use SOTA unsupervised techniques to"
    "develop a competive pipeline for the PKU kaggle competition."),
    install_requires=REQS,
)
