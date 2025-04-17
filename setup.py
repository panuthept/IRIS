from setuptools import setup, find_packages


with open("requirements.txt") as f:
    dependencies = [line for line in f]

setup(
    name='iris',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache-2.0 License',
    author='',
    author_email='',
    description='Improving Robustness of LLMs on Input Variations by Mitigating Spurious Intermediate States.',
    # python_requires='==3.11.4',
    # install_requires=dependencies
)