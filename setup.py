from setuptools import setup
#from TEACHING_DEEPLEARNING import __version__

setup(name='insung_lee_teaching_RL',
      #version=__version__,
      url='https://github.com/shipjobs/TEACHING_DEEPLEARNING',
      license='Apache-2.0',
      author='insung_lee',
      author_email='shippauljobs@gmail.com',
      description='TEACHING_DEEPLEARNING',
      packages=['TEACHING_DEEPLEARNING', ],
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      )

