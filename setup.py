from setuptools import setup, find_packages

setup(
    name="nlrs",
    version="0.1.0",
    author="Thomas R. Holy",
    author_email="th0ly96@gmail.com",
    description="This package provides Non-standard Linear Regression methods with Sparsity.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/trholy/nlrs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    license_files=('LICENSE',),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "cvxpy>=1.5.1",
        "scikit-learn>=1.2.2",
        "numpy>=1.24.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2",
            "ruff>=0.9.6",
            "asgl",
        ]
    },
    test_suite='pytest',
)
