from setuptools import setup

setup(
    name="epam_nlp",
    description="EPAM data science course nlp homework",
    packages=['epam_nlp'], install_requires=['scikit-learn', 'hmmlearn', 'numpy', 'pandas', 'seqlearn']
)
