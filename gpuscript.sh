#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<your_username> # required to send email notifcations - please replace <your_username> with your college login name or email address
source venv/bin/activate
. /vol/cuda/11.8.0/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
python3 -m StarMap train
