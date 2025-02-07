from setuptools import setup
import os
from glob import glob

package_name = 'percept'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    # package_dir={'': 'src'},
    # data_files=[
    #     ('share/ament_index/resource_index/packages',
    #         ['resource/' + package_name]),
    #     ('share/' + package_name, ['package.xml']),
    #     # Include all launch files
    #     (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    #     # Include all config files if you have any
    #     (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    # ],
    # install_requires=['setuptools'],
    # zip_safe=True,
    # maintainer='dev',
    # maintainer_email='dev@todo.todo',
    # description='The percept package',
    # license='TODO',
    # tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': [
    #         'real_pipeline = percept.real_pipeline:main',
    #         'static_tf_publisher = percept.utils.static_tf_publisher:main',
    #         # Add other executables here
    #     ],
    # },
    develop=True  # This enables development mode
)