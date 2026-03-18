# Small test config for fast experiments on OpenWebText subset

out_dir = 'test-ckpts'
dataset = 'openwebtext'
train_data_fraction = 0.01

# small model
n_layer = 2
n_head = 2
n_embd = 64

# training
max_iters = 10000
warmup_iters = 1000
lr_decay_iters = 1000
checkpoint_iters = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2000, 3000, 4000, 5000, 6000, 7000,
    8000, 9000, 9250, 9500, 9750, 1000
)

# eval / logging
eval_interval = 500
eval_iters = 50
log_interval = 100
wandb_run_name = 'gpt2micro-openwebtext'

# batch
batch_size = 128
block_size = 1024
gradient_accumulation_steps = 1
