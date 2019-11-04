import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='KFT',
     version='0.1',
     author="Robert Hu",
     author_email="robert.hu@stats.ox.ac.uk",
     description="Kernel Fried Tensor with Calibrated Variational Gravy",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/MrHuff/KernelFriedTensor",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )