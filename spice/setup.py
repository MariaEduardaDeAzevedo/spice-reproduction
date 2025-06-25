from setuptools import setup, find_packages


with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="spice_python",
    version="0.1",
    packages=find_packages(),
    description="A simple implementation of the metric SPICE in Python",
    author="Seu Nome",
    author_email="maria.silva@ccc.ufcg.edu.com.br",
    url="",
    install_requires=requirements,
)
