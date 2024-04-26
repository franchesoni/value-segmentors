#!/bin/bash
postfix=$1
maxtime=$2
# a100 or v100
gputype=$3
mem32gb=$4
gpus=${5:-1}

echo gpus=$gpus
[ "$gpus" = "1" ] && echo "Running single gpu" || echo "multigpu"

echo "Launching job ${postfix} for ${maxtime} gpu ${gputype} and ${mem32gb}"
if [ "$gputype" == "a100" ]; then
   optional_sbatch="#SBATCH --partition=gpu_p4"
elif [ "$gputype" == "v100" ]; then
   optional_sbatch=""
else
   echo "Invalid gpu type, should be a100 or v100"
   exit 1
fi

if [ "$mem32gb" == "bigram" ]; then
   optional_sbatch_mem="#SBATCH -C v100-32g"
   python_flag="--mem32gb"
elif [ "$mem32gb" == "lowram" ]; then
   optional_sbatch_mem=""
   python_flag=""
else
   echo "Invalid mem type, should be bigram or lowram, but is $mem32gb"
   exit 1
fi


sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${gpus}_${gputype}_${mem32gb}_${postfix}
#SBATCH --time=$maxtime
#SBATCH --output=logs/${gpus}_${gputype}_${mem32gb}_${postfix}.out
#SBATCH --error=logs/${gpus}_${gputype}_${mem32gb}_${postfix}.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=${gpus}
#SBATCH --gres=gpu:${gpus}
#SBATCH --hint=nomultithread
$(echo -e "$optional_sbatch")
$(echo -e "$optional_sbatch_mem")

# Set up pre-defined Python env
module purge
module load pytorch-gpu/py3/2.2.0
module load git/2.39.1
cd ${SCRATCH}/value-segmentors/train_ade20k/

set -x
if [ "$gpus" = "1" ]; then
   echo "Running single gpu"
   srun python -u train_script.py --clean --batch_size=16 --epochs=50 --val_every_pct=0.05 --tag="_${postfix}" 
else
   echo "Running multigpu isn't supported yet"
   exit 1
   srun torchrun --standalone --nproc_per_node=gpu single_script.py /linkhome/rech/genwnn01/uyz17rc/tuto/data/processed/crops/ --epochs=8 --tag="_${postfix}" --val_size=10 --multigpu --data_limit=1000
fi


EOT

# e.g. run with `bash launch_slurm.sh dev 00:05:00 v100 bigram``