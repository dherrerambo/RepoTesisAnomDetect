import pathlib, os
from setuptools import setup, find_packages

root_path = pathlib.Path(__file__).parent

root_path.cwd()

print(os.getcwd())

PACKAGE_NAME = 'lib_anomaly'
VERSION = '1.1'
DESCRIPTION = 'helpers for anomaly detection tesis'
# LONG_DESCRIPTION = (HERE / "README.md").read_text()
# LONG_DESC_TYPE = "text/markdown"

print(f"{os.getcwd()=}")

try:
      INSTALL_REQUIRES = open(
                  root_path.joinpath("requirements.txt")
                  , mode="r", encoding="utf8"
            ).read().split("\n")
except:
      INSTALL_REQUIRES = []

setup(
      name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      # long_description=LONG_DESCRIPTION,
      # long_description_content_type=LONG_DESC_TYPE,
      author='Diego Herrera Malambo',
      license='Apache License 2.0',
      author_email='dherrerambo@gmail.com',
      url='https://github.com/dherrerambo/TesisAnomDetect',
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
)
