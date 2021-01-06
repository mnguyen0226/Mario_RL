# Mario_RL_Adept
About: Implement Mario DQN from Pytorch Tutorial with Adept Framework @Heron Systems
Credit: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
Framework: https://github.com/heronsystems/adeptRL
Wiki@Heron: http://192.168.1.30/

=======================================================================================================
Fix in install adeptRL!

    adeptRL: git clone repo on Heron Systems
    After git clone, git pull adept, do:
    pip install -U ray[tune]
    conda install protobuf

=======================================================================================================

***** Training command local or gpu *****

python adept_mario_script.py --agent AdeptMarioAgent --nb-env 2 --env SuperMarioBros-1-1-v0 --net3d AdeptMarioNet --netbody Linear

***** Steps *****

1/ Run Mario Pytorch Code on local machine
2/ Divide Mario Code into sub-sections
3/ Implement Mario Pytorch on Adept framework and train with GPU 
   + Implement Environment
   + Implement Network Submodule3D Four-Conv
   + Implement Replay
   + Implement Agent
   + Implement Actor - Learner
4/ Train on GPU Banshee @ Heron Systems
   
=========================================================================================================
***** Notes *****

1/ How to start a Python Project? (Assume working on Pycharm + Anaconda + GPU)
* Terminal:
   conda create -n cartpole python=3.8
   conda activate cartpole
   -- copy setup.py and adjust accordingly with libraries to avoid pip install
   python setup.py develop
* Pycharm:
   File => Setting => Project: cartpole
   Python Interpreter => "Circle symbol" => Existing environment => bin => python
   OK

2/ How to train/run on GPU/Banshee/Siren
   a/ Copy training files to GPU:
      rsync -r /home/minhtnguyen/Documents/myproject/Tutorials/CartPole_DQN_Project /media                                                    
      /banshee/users/minh/
   b/ Log in to GPU:
      ssh heron@banshee
      PW: Check google note
   c/ on Heron@Banshee, do cd /media and go to working directory containing setup.py
      Terminal:
         conda create -n cartpole python=3.8
         conda activate cartpole
         -- copy setup.py and adjust accordingly with libraries to avoid pip install
         python setup.py develop
      Go to the .py file for training (for ex: Mario_RL/marioRL/Adept/script/adept_main.py)
      Execute the training command and wait for results every 6000000 steps in "checkpoints" directory

3/ How to run on TMUX over night? (assume already copy files to /media/banshee with rsync -r)
   ssh heron@banshee
   PW: Check google note
   cd /media => to your training file or directory
   
   tmux new 
   Ctrl-B + D to exit but not stop, remember number #
   tmux attach -t #
   then run the train command
   
   TO EXIT WITHOUT STOP: Ctrl-B + D
   TO STOP: CRTL-D or type exit
   
========================================================================================================
