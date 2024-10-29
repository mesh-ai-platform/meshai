# setup.py

from setuptools import setup, find_packages

setup(
    name='meshai',
    version='0.1.6',
    description='MeshAI SDK for decentralized AI model development',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/robbie/meshai-sdk',
    author='Robbie (Ravi Tiwari)',
    author_email='your.email@example.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'transformers',
        'pandas',
        'scikit-learn',
        'numpy',
        'joblib',
        'Pillow',
        'PyPDF2',
        'cryptography',  # Added for security features
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
