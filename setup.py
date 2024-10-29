# setup.py

from setuptools import setup, find_packages

setup(
    name='meshai',  # This is the name of your package
    version='0.1.3',
    description='MeshAI SDK for decentralized AI model development',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mesh-ai-platform/meshai',  # Replace with your repository URL
    author='Robbie Tiwari',
    author_email='rtiwariops@gmail.com',
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
        'Pillow'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
