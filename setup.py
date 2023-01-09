from setuptools import setup

setup(
    name="jaxtomo",
    version="1.0",
    description="Tomographic projector in JAX.",
    author="Clemens Schmid",
    author_email="clem.schmid@tum.de",
    packages=["jaxtomo"],
    install_requires=["numpy", "scipy", "jax"],
    license="MIT",
)
