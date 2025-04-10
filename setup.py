from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="benchmark-prototype-sse",
    version="0.1.0",
    author="Group-24",
    description="Modular MLPerf Inference Framework with Vision Extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sgrad20Delft/inference",
    packages=find_packages(
        include=[
            "vision*",
            "tools*",
            "automotive*",
            "recommendation*",
            "language*",
            "loadgen*",
            "metrics*",
        ]
    ),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vision-infer=vision.general.generalized_main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
