from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="your_package_name",
    version="0.1.0",
    description="A brief description of your package",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/yourrepo",
    packages=find_packages(),
    install_requires=requirements,  # Load dependencies dynamically
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.8",
)

