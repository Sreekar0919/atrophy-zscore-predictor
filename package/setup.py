from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="atrophy_predictor",
    version="0.1.0",
    author="RIDR Team",
    author_email="your.email@example.com",
    description="A tool for predicting brain atrophy patterns from clinical notes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/atrophy_predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.8.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "python-dotenv>=0.19.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
    },
) 