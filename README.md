# percept

![alt text](imgs/banner.png)


## Debug Fixes

### OmegaConf Error #1
```
pkg_resources.extern.packaging.requirements.InvalidRequirement: .* suffix can only be used with `==` or `!=` operators
    PyYAML (>=5.1.*)
```
is related to the omegaconf package's METADATA file and NOT pyyaml. The authors of the omegaconf package stopped maintaining it after version 2.0.6 and this is a PEP-related bug (non-standard dependency specifier) that arises when a description generator (?) is invoked. Luckily there is an easy workaround.

`<NAME-OF-YOUR-PERACT-CONDA-ENVIRONMENT>` = peract
1. Navigate to package site:
```
cd ~/miniconda3/envs/<NAME-OF-YOUR-PERACT-CONDA-ENVIRONMENT>/lib/python3.8/site-packages/omegaconf-2.0.6.dist-info
```
2. Edit the METADATA file and change:

```
Requires-Dist: PyYAML (>=5.1.*)
to
Requires-Dist: PyYAML (>=5.1)
```