from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='OnePiecePredictor',
    version='0.1',
    description='Hyper Paramter Tuning and Models performance',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='Vineel Kurma',
    author_email='vineel.prince7@gmail.com',
    keywords=['OnePiecePredictor'],
    url='https://github.com/vkurma/OnePiecePredictor',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

install_requires = [
    'numpy',
    'pandas',
    'scikit-learn',
    'xgboost',
    'catboost',
    'category_encoders',
    'imblearn',
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)