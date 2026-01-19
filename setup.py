from setuptools import find_packages, setup

setup(
    name="Hybrid-Video-Recommender-System",
    version="0.1.0",
    author="Idowu Thomas",
    author_email="ayanfeoluwadegoke@gmail.com",
    description="Hybrid recommender system using content-based and collaborative filtering",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[],
)


