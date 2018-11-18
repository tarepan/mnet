from setuptools import setup, find_packages
setup(
    name="mnet",
    version="0.3.0",
    packages=find_packages(),
    install_requires = [
        "numpy",
        "scipy",
        "functional",
        "librosa",
        "pyworld"
    ]
)
