# Mario_RL_Adept
About: Implement Mario DQN from Pytorch Tutorial with Adept Framework @Heron Systems

Credit: 

    https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        
Framework: https://github.com/heronsystems/adeptRL
    After conda create, beside doing the "python setup.py develop" in gamebreaker folder, go to adeptRL that you download and do "pip install -e .[all]"

Wiki@Heron: http://192.168.1.30/

=============================================================================

Fix in install adeptRL!

    adeptRL: git clone repo on Heron Systems
    After git clone, git pull adept, do:
    pip install -U ray[tune]
    conda install protobuf

=============================================================================

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
   
=============================================================================

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
   
=============================================================================

FILE STRUCTURE EXPLAINED:

1/ Original Pytorch Mario DQN Tutorial: Mario_RL/mario.py

2/ Module-divided Mario DQN:
Mario_RL/marioRL/Split
    - module 
    
        - agent_dqn.py
        
        - logger.py
        
        - marioNet.py
        
        - preprocess.py
        
    - scripts
    
        - checkpoints (saved trained agents)
        
        - main.py 
        
    (RUN: python ./main.py to execute the training)

Mario_RL/marioRL/Adept

    - modules
    
        - adept_actor.py
        
        - adept_agent.py
        
        - adept_env.py
        
        - adept_learner.py
        
        - adept_net.py
        
        - adept_replay.py
        
    - script
    
        - adept_main.py
        
    (RUN: python adept_mario_script.py --agent AdeptMarioAgent --nb-env 2 --env SuperMarioBros-1-1-v0 --net3d AdeptMarioNet --netbody Linear)
    
=============================================================================

How to check trained agent on banshee GPU @ Heron?
    ssh heron@banshee - PW: check phone
    
    cd /tmp/adept_logs/SuperMarioBros-1-1-v0/Local_AdeptMarioAgent_Linear_2021-01-05-18-23-00
    
    Then copy that dir to /media/banshee/users/minh/mario_test/
    
    go to local machine and cd to /media/banshee/users/minh/mario_test/SuperMarioBros (...)

    cd to latest trained log
    
    RUN: python -m adept.scripts.render --logdir . --epoch 2e6

=============================================================================

How to Run Tensorboard on trained model?
    ssh heron@banshee - PW: check phone
    
    conda activate marioRL
    
    cd /tmp/adept_logs/SuperMarioBros-1-1-v0/Local_AdeptMarioAgent_Linear_2021-01-05_18-23-00 
    
    tensorboard --logdir . --bind_all
    
    
    
