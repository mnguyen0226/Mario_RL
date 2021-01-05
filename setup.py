from setuptools import setup, find_packages
# https://github.com/kennethreitz/setup.py/blob/master/setup.py
with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="mario_RL",
    version="0.0.1",
    author="Minh Nguyen",
    author_email="mnguyen0226@vt.edu",
    description="Mario Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license="GNU",
    python_requires=">=3.5.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "pandas",
        "gym",
        "torch",
        "Pillow <= 7.2.0",
        "matplotlib",
        "torchvision",
        "nes_py",
        "gym_super_mario_bros",
        "adeptRL>=0.2.0"
    ],
    include_package_data=True,
)
