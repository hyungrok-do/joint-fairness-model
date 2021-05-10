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

for p in 50 100 200 400 600 800 1000 1200 1400 1600 1800 2000
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
           --name "sim4_${p}_${target_seed}" \
           --seed $target_seed \
           --p $p
  done
done

for t in 0 2 4 6 8 10 12 14 16 18 20
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
            --name "sim1b_${t}_${seed}" \
            --seed $seed \
            --r 0 \
            --t $t
  done
done

for b in 0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
            --name "sim2b_${b}_${seed}" \
            --seed $seed \
            --b $b
  done
done

for n in 500 800 1100 1400 1700 2000 2300 2600 2900 3200 3500
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
           --name "sim3b_${n}_${target_seed}" \
           --seed $target_seed \
           --n1 $n
  done
done

for p in 50 100 200 400 600 800 1000 1200 1400 1600 1800 2000
do
  for seed in {0..19}
  do
    python experiments-simulation.py \
           --name "sim4b_${p}_${target_seed}" \
           --seed $target_seed \
            --p $p \
            --q $((p * 3 / 10)) \
            --r $((p * 3 / 20))
  done
done