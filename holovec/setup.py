from setuptools import setup, find_packages

setup(
    name="holovec",
    version="0.1.0",
    description="A Hyperdimensional Computing (HDC) Library",
    author="Gemini CLI",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
    ],
    python_requires=">=3.7",
)
