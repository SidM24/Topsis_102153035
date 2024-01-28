from setuptools import setup, find_packages

setup(
    name='Topsis_102153035',  # <--- Updated name
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'Topsis_102153035 = Topsis_102153035._main_:main'
        ],
    },
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown'
)