from setuptools import setup, find_packages

setup(
    name="llava-prune",
    version="0.1.0",
    description="A package for pruning and inference with LLaVA models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "Pillow",
        "requests",
        "tqdm",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "llava-prune=scripts.run_pruning:main",
        ],
    },
)
