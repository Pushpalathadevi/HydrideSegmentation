from setuptools import find_packages, setup

setup(
    name='hydride-segmentation',
    version='1.2.0',
    description='Toolkit for zirconium hydride segmentation and analysis',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # Templates and static assets are served by the intranet web app and must
        # ship with the installed package, not only from a source checkout.
        'hydride_segmentation.web': [
            'templates/*.html',
            'static/css/*.css',
            'static/js/*.js',
            'static/img/*.svg',
            # Vendored KaTeX. The fonts must ship too, or the help page renders
            # its mathematics in a fallback face on an air-gapped host.
            'static/vendor/katex/*.css',
            'static/vendor/katex/*.js',
            'static/vendor/katex/fonts/*.woff2',
            'static/vendor/katex/LICENSE',
            'static/vendor/katex/README.md',
        ],
    },
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'hydride-gui=hydride_segmentation.gui:main',
            'hydride-gui-qt=hydride_segmentation.qt_gui:main',
            'hydride-orientation=hydride_segmentation.hydride_orientation_analyzer:main',
            'segmentation-eval=hydride_segmentation.segmentation_evaluator:main',
            'microseg-cli=scripts.microseg_cli:main',
            'microseg-web=scripts.run_web_server:main',
            'microseg-phase-gate=scripts.run_phase_gate:main',
            'microseg-benchmark-suite=scripts.hydride_benchmark_suite:main',
            'prep-dataset=src.microseg.data_preparation.cli:main',
        ]
    },
)
