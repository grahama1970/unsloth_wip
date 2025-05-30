from setuptools import setup, find_packages

setup(
    name="unsloth",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "unsloth-cli=unsloth.cli.main_minimal:app",
        ],
    },
    install_requires=[
        "typer>=0.16.0",
        "rich>=14.0.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
    ],
)
