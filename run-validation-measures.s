#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=ValidMeasure
#SBATCH --output=./logs/VM_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your@mail
#SBATCH -a 0-659

module load anaconda3/2020.07

r_dict=(0 4 8 12 16 20 24 28 32 36 40)
b_dict=(0 -2.5 -5 -7.5 -10 -12.5 -15 -17.5 -20 -22.5 -25)
n_dict=(50 75 100 125 150 175 200 225 250 275 300)

for scenario in 0 1 2
do
    for seed in {0..19}
    do
        for set_id in {0..10}
        do
            let task_id=$set_id+$seed*11+$scenario*220
            if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
                target_scenario=$scenario
                target_seed=$seed
                target_set_id=$set_id
            fi
        done
    done
done

echo $SLURM_ARRAY_TASK_ID
echo $target_scenario
echo $target_seed
echo $target_set_id

if [ "$target_scenario" -eq 0 ]
then
    r=${r_dict[target_set_id]}
    python experiments-validation-measures.py \
            --name "sim1_${r}_${target_seed}" \
            --seed $target_seed \
            --r $r
elif [ "$target_scenario" -eq 1 ]
then
    b=${b_dict[target_set_id]}
    python experiments-validation-measures.py \
            --name "sim2_${b}_${target_seed}" \
            --seed $target_seed \
            --b $b
elif [ "$target_scenario" -eq 2 ]
then
    n=${n_dict[target_set_id]}
    python experiments-validation-measures.py \
            --name "sim3_${n}_${target_seed}" \
            --seed $target_seed \
            --n2 $n
fi