import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SEAVIEW",
    version="0.0.1",
    author="Rose Awen Brindle",
    author_email="rob87@aber.ac.uk",
    description="Robotic fish position tracking package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RosesHaveThorns/SEAVIEW",
    project_urls={
        "Bug Tracker": "https://github.com/RosesHaveThorns/SEAVIEW/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)