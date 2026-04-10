from setuptools import setup, find_packages


def _load_requirements(path: str) -> list[str]:
    requirements: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                nested = line[3:].strip()
                requirements.extend(_load_requirements(nested))
                continue
            requirements.append(line)
    return requirements


requirements = _load_requirements("requirements.txt")

setup(
    name='hydride-segmentation',
    version='0.22.0',
    description='Toolkit for zirconium hydride segmentation and analysis',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'hydride-gui=hydride_segmentation.gui:main',
            'hydride-gui-qt=hydride_segmentation.qt_gui:main',
            'hydride-orientation=hydride_segmentation.hydride_orientation_analyzer:main',
            'segmentation-eval=hydride_segmentation.segmentation_evaluator:main',
            'package-corrections-dataset=scripts.package_corrections_dataset:main',
            'microseg-cli=scripts.microseg_cli:main',
            'microseg-phase-gate=scripts.run_phase_gate:main',
            'microseg-benchmark-suite=scripts.hydride_benchmark_suite:main',
            'prep-dataset=src.microseg.data_preparation.cli:main',
        ]
    },
)
