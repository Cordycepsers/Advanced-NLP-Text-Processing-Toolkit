from setuptools import setup, find_packages

setup(
    name='advanced-nlp-text-processor',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'spacy>=3.7.4',
        'textblob>=0.17.1',
        'transformers>=4.39.3',
        'tensorflow>=2.15.0',
        'gensim>=4.3.2',
        'scikit-learn>=1.3.2',
        'nltk>=3.8.1',
        'pandas>=2.2.1',
        'numpy>=1.26.3'
    ],
    author='maja@whattookyousolong.com',
    author_email='maja@whattookyousolong.com',
    description='Advanced NLP Text Processing Toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cordyceps/advanced-nlp-text-processor',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)