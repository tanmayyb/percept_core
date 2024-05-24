# root commands
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential
sudo apt install git

# activate conda environment
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate


# setup workspace
mkdir -p ~/ws_percept && cd ~/ws_percept
export PERCEPT_ROOT=$(pwd) # export PERCEPT_ROOT="/home/dev/ws_percept"


# clone peract
git clone https://github.com/peract/peract.git
# conda create --name peract python=3.8  && conda activate peract 

# setup virtualenv
sudo apt install python3-virtualenv
virtualenv -p $(which python3.8) --system-site-packages peract_env  
source peract_env/bin/activate
pip install --upgrade pip
cd peract
export PERACT_ROOT=$(pwd)   # export PERACT_ROOT="/home/dev/ws_percept/peract"


# download and extract coppelia sim
mkdir -p ~/ws_coppelia && cd ~/ws_coppelia
export WS_COPPELIA=$(pwd) # export WS_COPPELIA="/home/dev/ws_coppelia"
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz
tar -xJf CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Player_V4_1_0_Ubuntu20_04 coppeliasim
rm -rf  CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz
cd coppeliasim 
export COPPELIASIM_ROOT=$(pwd) # export COPPELIASIM_ROOT="/home/dev/ws_coppelia/coppeliasim"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT


# setup PyRep
cd ~/ws_coppelia
git clone https://github.com/stepjam/PyRep
cd PyRep
pip install -r requirements.txt
pip install .
export PYREP_ROOT=$(pwd)    # export PYREP_ROOT="/home/dev/ws_coppelia/PyRep"


#setup libs
cd $PERCEPT_ROOT
mkdir libs && cd libs
export PERCEPT_LIBS=$(pwd)  # export PERCEPT_LIBS="/home/dev/ws_percept/libs"

# setup RLBENCH
git clone -b peract https://github.com/MohitShridhar/RLBench.git 
cd RLBench
pip install -r requirements.txt
python setup.py develop
export RLBENCH_ROOT=$(pwd)  # export RLBENCH_ROOT="/home/dev/ws_percept/libs/RLBench"

# setup YARR
cd $PERCEPT_LIBS
git clone -b peract https://github.com/MohitShridhar/YARR.git # note: 'peract' branch
cd YARR
pip install -r requirements.txt
python setup.py develop
export YARR_ROOT=$(pwd)     # export YARR_ROOT="/home/dev/ws_percept/libs/YARR"


# setup peract
cd $PERACT_ROOT
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
python setup.py develop


# setup complete!