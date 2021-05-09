for r in 0 4 8 12 16 20 24 28 32 36 40
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
            --name "sim1_${r}_${seed}" \
            --seed $seed \
            --r $r
  done
done

for b in 0 -2.5 -5 -7.5 -10 -12.5 -15 -17.5 -20 -22.5 -25
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
            --name "sim2_${b}_${seed}" \
            --seed $seed \
            --b $b
  done
done

for n in 50 75 100 125 150 175 200 225 250 275 300
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
           --name "sim3_${n}_${target_seed}" \
           --seed $target_seed \
           --n2 $n
  done
done
