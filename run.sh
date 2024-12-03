for random_seed in 2020 2021 2022 2023 2024 # 2024
do
    python -u main_exp.py --dataset AD --in_features 16 --n_classes 2 --seq_len 256 --dropout 0.2 --dropout_patch 0.1 --lr 5e-3 --weight_decay 0.001 --batch_size 128 --epochs 20 --cal_fraction 0.0 --seed $random_seed
    python -u main_exp.py --dataset Epilepsy --in_features 1 --n_classes 5 --seq_len 178 --dropout 0.2 --dropout_patch 0.1 --lr 5e-3 --weight_decay 0.001 --batch_size 128 --epochs 20 --cal_fraction 0.0 --seed $random_seed
    python -u main_exp.py --dataset TDBrain --in_features 33 --n_classes 2 --seq_len 256 --dropout 0.2 --dropout_patch 0.1 --lr 5e-3 --weight_decay 0.001 --batch_size 128 --epochs 20 --cal_fraction 0.0 --seed $random_seed
    python -u main_exp.py --dataset PTBXL10 --in_features 12 --n_classes 5 --seq_len 2500 --dropout 0.2 --dropout_patch 0.1 --lr 5e-3 --weight_decay 0.001 --batch_size 128 --epochs 10 --cal_fraction 0.0 --seed $random_seed
    python -u main_exp.py --dataset ADFD --in_features 19 --n_classes 3 --seq_len 256 --dropout 0.2 --dropout_patch 0.1 --lr 5e-3 --weight_decay 0.001 --batch_size 128 --epochs 10 --cal_fraction 0.0 --seed $random_seed
    python -u main_exp.py --dataset PTBXL --in_features 12 --n_classes 5 --seq_len 250 --dropout 0.2 --dropout_patch 0.1 --lr 5e-3 --weight_decay 0.001 --batch_size 128 --epochs 10 --cal_fraction 0.0 --seed $random_seed
    python -u main_exp.py --dataset sleepEDF --in_features 1 --n_classes 5 --seq_len 3000 --dropout 0.2 --dropout_patch 0.2  --lr 2.5e-3 --weight_decay 0.001 --batch_size 128 --epochs 20 --cal_fraction 0.0 --seed $random_seed
done

