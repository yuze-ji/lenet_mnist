# lenet/config.py
# Default hyperparameters and paths. Override any of these via main.py CLI flags.

batch_size = 64
epochs_pretrain = 5
epochs_finetune = 3
learning_rate = 0.01
momentum = 0.9

data_dir = "data"
checkpoint_dir = "checkpoints"
log_dir = "logs"
