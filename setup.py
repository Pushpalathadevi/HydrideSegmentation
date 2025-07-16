from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name='hydride-segmentation',
    version='0.1.0',
    description='Toolkit for zirconium hydride segmentation and analysis',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'hydride-gui=hydride_segmentation.gui:main',
            'hydride-orientation=hydride_segmentation.hydride_orientation_analyzer:main',
            'segmentation-eval=hydride_segmentation.segmentation_evaluator:main',
        ]
    },
)
