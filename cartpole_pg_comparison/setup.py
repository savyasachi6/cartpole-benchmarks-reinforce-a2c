from setuptools import setup, find_packages

setup(
    name="cartpole-pg-comparison",
    version="0.1.0",
    description="implementation comparing policy gradient methods on CartPole-v1",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0"
    ],
    python_requires=">=3.8",
)
