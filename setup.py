from setuptools import setup, find_packages

setup(
    name="dod_dangerousObjectDetector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        'ultralytics',
        'opencv-python'
    ],
)