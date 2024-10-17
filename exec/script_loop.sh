
n_gpu=0

# Linear Regression  ###################################################################################################
b=30
T=15
#python scripts/train.py --config configs/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "LR_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu
python scripts/train.py --config configs/base_loop.yaml --model.n_layer 1  --training.curriculum.loops.start 15  --training.curriculum.loops.end 30 --training.n_loop_window 15  --wandb.name "LR_loop_L1_endsB30_T15"  --gpu.n_gpu 0
# Sparse Linear Regression  ############################################################################################
b=20
T=10
#python scripts/train.py --config configs/sparse_LR/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "SparseLR_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu

# Decision Tree ########################################################################################################
#python scripts/train.py --config configs/decision_tree/base_loop.yaml --model.n_layer 1 --training.curriculum.loops.start 15 --training.curriculum.loops.end 70 --training.n_loop_window 15 --wandb.name "DT_loop_L1_endsb70_T15" --gpu.n_gpu 0
b=70
T=15
#python scripts/train.py --config configs/decision_tree/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "DT_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu

# ReLU2NN  #############################################################################################################
b=12
T=5
#python scripts/train.py --config configs/relu_2nn_regression/base_loop.yaml --model.n_layer 1 --training.curriculum.loops.start 5 --training.curriculum.loops.end 12 --training.n_loop_window 5 --wandb.name "relu2nn_loop_L1_endsb12_T5" --gpu.n_gpu 0

#python scripts/train.py --config configs/relu_2nn_regression/base_loop.yaml \
#    --model.n_layer 1 \
#    --training.curriculum.loops.start $T \
#    --training.curriculum.loops.end $b \
#    --training.n_loop_window $T \
#    --wandb.name "relu2nn_loop_L1_ends{$b}_T{$T}" \
#    --gpu.n_gpu $n_gpu
