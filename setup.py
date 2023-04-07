from setuptools import setup, find_packages


setup(
    name="pymcda",
    version="0.1.0",
    description="Python library for Multi-Criteria Decision Analysis",
    package_dir={"": "pymcda"},
    packages=find_packages("pymcda"),
    package_data={"pymcda": ["py.typed"]},
    author="Bartosz Stachowiak",
    author_email="sbartekt@op.pl",
    license="MIT",
    install_requires=[
        "numpy",
        "pulp",
    ],
    python_requires=">=3.8",
)
