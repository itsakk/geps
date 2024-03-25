from setuptools import find_packages, setup

setup(
    name="fuels",
    version="0.0.1",
    description="Package for learning to adapt parametric PDEs",
    author="Armand Kassai",
    author_email="kassai@isir.upmc.fr",
    install_requires=[
        "einops",
        "hydra-core",
        "wandb",
        "torch",
        "pandas",
        "matplotlib",
        "scipy",
        "h5py",
        "torchdiffeq",
    ],
    package_dir={"fuels": "fuels"},
    packages=find_packages(),
)