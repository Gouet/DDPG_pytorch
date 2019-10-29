# DDPG_pytorch

The DDPG algortithme has been tested with the Pendulum GYM env.
Using Pytorch 1.3.0

**Launch for traning**

python train.py --scenario=Pendulum-v0 --saved-episode 50

**Launch for testing**

python train.py --scenario=Pendulum-v0 --eval --load-episode-saved=160

# Optional parameters

| Commands  | Descriptions |
| ------------- | ------------- |
| --scenario [string] | Gym environments  |
| --eval  | Load a default AI and render it  |
| --load-episode-saved [int] | Load an AI  |
| --saved-episode [int] | Save episode at each [EPISODE] times  |
| --batch-size [int] | Batch size for the training  |
| --max-episode [int] | Episodes end  |
| --gamma [float] | Discount factor  |
| --tau [float] | For the targets network and copy data  |
