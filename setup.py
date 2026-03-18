from setuptools import setup, find_packages

setup(
    name="icrl",
    version="0.1.0",
    description="Interpretable Continuous Reinforcement Learning in Robotics",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "gymnasium",
        "hydra-core",
        "omegaconf",
        "wandb",
        "matplotlib",
        "tensorboard",
    ],
    extras_require={
        "mujoco": ["mujoco", "dm-control"],
        "dev": ["pytest", "black", "ruff", "pre-commit"],
    },
)
