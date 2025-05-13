from setuptools import find_packages, setup

setup(
    name="stock_prediction",
    version="0.1.0",
    packages=find_packages(),
    description="Classical vs Quantum ML for Stock Price Prediction",
    author="Simone Meli",
    author_email="simone.meli25@outlook.it",
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "scikit-learn",
        "tensorflow",
        "matplotlib",
        "seaborn",
        "jupyter",
        "tqdm",
        "python-dotenv",
    ],
    python_requires=">=3.8",
) 