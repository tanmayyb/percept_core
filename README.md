# percept

![alt text](imgs/banner.png)


## Debug Fixes

### Error with OmegaConf
Error:
```
pkg_resources.extern.packaging.requirements.InvalidRequirement: .* suffix can only be used with `==` or `!=` operators
    PyYAML (>=5.1.*)
```
Solution:
```
cd ~/miniconda3/envs/peract/lib/python3.8/site-packages/omegaconf-2.0.6.dist-info
```
and edit `METADATA` file and change
```
Requires-Dist: PyYAML (>=5.1.*)
to
Requires-Dist: PyYAML (>=5.1)
```