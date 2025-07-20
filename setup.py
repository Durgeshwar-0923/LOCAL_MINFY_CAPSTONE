from setuptools import setup, find_packages

setup(
    name="instilit_ml_pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # requirements are in requirements.txt
    ],
    author="Your Name",
    description="Global Salary Intelligence ML Pipeline",
)
