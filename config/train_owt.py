# Config for experiments on OpenWebText

out_dir = 'ckpts-30000k'
dataset = 'openwebtext'
train_data_fraction = 1.0

# small model
n_layer = 12
n_head = 12
n_embd = 768

# training
max_iters = 30000
warmup_iters = 2000
lr_decay_iters = 2000
checkpoint_iters = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
    21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 28250, 28500,
    28750, 29000, 29250, 29500, 29750, 30000,
)
learning_rate = 1e-5

# eval / logging
eval_interval = 1000
eval_iters = 200
log_interval = 10
wandb_run_name = 'gpt2-openwebtext'

# batch size 64 batch x 1024 block x 8 accum steps x 2 GPU = 1M tokens
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 8
