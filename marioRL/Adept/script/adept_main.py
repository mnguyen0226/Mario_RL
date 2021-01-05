"""
    adept main script
    adeptRL: git clone repo on Heron Systems
    After git clone, git pull adept, do:
     pip install -U ray[tune]
    conda install protobuf
    1/ How to run the environment
    python adept_main.py --env SuperMarioBros-1-1-v0 --net3d AdeptMarioNet
        --manager SimpleEnvManager --nb-env 2

    IMPLEMENT:
        env
        replay
        net
        agent

    * How to run Tmux?
    ssh heron@banshee => HeronR&D
    tmux new
    tmux attach -t 13
    => Run your session

    Tmux -t 15: for testing purposes

"""
from adept.scripts.local import parse_args, main
from marioRL.Adept.modules.adept_env import AdeptMarioEnv
from marioRL.Adept.modules.adept_net import AdeptMarioNet

print("adept_main.py is running")

if __name__ == '__main__':
    import adept
    adept.register_env(AdeptMarioEnv)
    adept.register_submodule(AdeptMarioNet)
    main(parse_args())