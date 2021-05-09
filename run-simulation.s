#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=SIM
#SBATCH --output=./logs/JFM_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your@mail
#SBATCH -a 0-1319

module load anaconda3/2020.07

r_dict=(0 4 8 12 16 20 24 28 32 36 40)
b_dict=(0 -2.5 -5 -7.5 -10 -12.5 -15 -17.5 -20 -22.5 -25)
n_dict=(50 75 100 125 150 175 200 225 250 275 300)

t_dict=(0 2 4 6 8 10 12 14 16 18 20)
b_dict2=(0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25)
n_dict2=(500 800 1100 1400 1700 2000 2300 2600 2900 3200 3500)


for scenario in 0 1 2 3 4 5
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
    python experiments-simulation.py \
            --name "sim1_${r}_${target_seed}" \
            --seed $target_seed \
            --r $r
elif [ "$target_scenario" -eq 1 ]
then
    b=${b_dict[target_set_id]}
    python experiments-simulation.py \
            --name "sim2_${b}_${target_seed}" \
            --seed $target_seed \
            --b $b
elif [ "$target_scenario" -eq 2 ]
then
    n=${n_dict[target_set_id]}
    python experiments-simulation.py \
            --name "sim3_${n}_${target_seed}" \
            --seed $target_seed \
            --n2 $n
elif [ "$target_scenario" -eq 3 ]
then
    t=${t_dict[target_set_id]}
    python experiments-simulation.py \
            --name "sim1b_${t}_${target_seed}" \
            --seed $target_seed \
            --r 0 \
            --t $t
elif [ "$target_scenario" -eq 4 ]
then
    b=${b_dict2[target_set_id]}
    python experiments-simulation.py \
            --name "sim2b_${b}_${target_seed}" \
            --seed $target_seed \
            --b $b
elif [ "$target_scenario" -eq 5 ]
then
    n=${n_dict2[target_set_id]}
    python experiments-simulation.py \
            --name "sim3b_${n}_${target_seed}" \
            --seed $target_seed \
            --n1 $n
fi