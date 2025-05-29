import wandb
import train_vector128   # import your train.py main method
# 
# Optional: login with API key or let wandb.cli handle it
wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

# Load sweep config
import yaml
with open("src/MLP/sweep.yaml") as f:
    sweep_config = yaml.safe_load(f)

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="Bachelor_projekt")

# Start agent
wandb.agent(sweep_id, function=train_vector128.main, count=16)
