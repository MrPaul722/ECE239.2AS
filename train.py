"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig

def solver(model_name):
    # Initialize the model
    if model_name == "bigram":
        config = BigramConfig
        model = BigramLanguageModel(config)
    elif model_name == "minigpt":
        config = MiniGPTConfig
        model = MiniGPT(config)
    else:
        raise ValueError("Invalid model name")
    
    # Load the dataset
    train_dataset = TinyStoriesDataset(
        config.path_to_data,
        mode="train",
        context_length=config.context_length,
    )
    eval_dataset = TinyStoriesDataset(
        config.path_to_data, mode="test", context_length=config.context_length
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.batch_size, pin_memory=True
    )

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print number of parameters in the model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))

    # Initialize wandb if you want to use it
    if config.to_log:
        wandb.init(project="dl2_proj3")

    # Create the save path if it does not exist
    if not Path.exists(config.save_path):
        Path.mkdir(config.save_path, parents=True, exist_ok=True)

    ### ==================== START OF YOUR CODE ==================== ###
    """
    You are required to implement the training loop for the model.

    The code below is a skeleton for the training loop, for your reference. 
    You can fill in the missing parts or completely set it up from scratch.

    Please keep the following in mind:
    - You will need to define an appropriate loss function for the model.
    - You will need to define an optimizer for the model.
    - You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
    - It is recommended that you save the model weights every `config.save_iterations` iterations. You can also just save the model with the best training loss.

    NOTE : 
    - Please check the config file to see the different configurations you can set for the model.
    - The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
    not a required part of the assignment. 
    - Feel free to experiment with the parameters and I would be happy to talk to you about them if interested.
    """

    ### ========= TODO : START ========= ###
    # Define the loss function
    debug = True
    loss = nn.CrossEntropyLoss()
    if debug:
      print("Loss function initialized.")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if debug:
      print("Optimizer initialized.")
    ### ======== TODO : END ========= ###

    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2
        )
        print("Scheduler initialized.")
    model.train()
    model.to(device)
    print("Starting training loop...")

    for i, (context, target) in enumerate(train_dataloader):

        train_loss = None # You can use this variable to store the training loss for the current iteration
        ### ======== TODO : START ========= ###
        # Do the forward pass, compute the loss, do the backward pass, and update the weights with the optimizer.
        
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(context)  # (bsz, seq_len, vocab_size) or (bsz, vocab_size)
        logits = outputs.view(-1, outputs.size(-1))
        targets_flat = target.view(-1)
        loss_value = loss(logits, targets_flat)
        loss_value.backward()

        if config.to_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        train_loss = loss_value.item()
        
        ### ======== TODO : END ========= ###

        if config.scheduler:
            scheduler.step()

        del context, target # Clear memory

        if i % config.log_interval == 0:
            print(f"Running evaluation at iteration {i}...", flush=True)
            model.eval()
            eval_loss = None # You can use this variable to store the evaluation loss for the current iteration
            ### ======== TODO : START ========= ###
        
            
            total_eval = 0.0
            count = 0
            
            with torch.no_grad():
                for idx, (c, t) in enumerate(eval_dataloader):
            
                    if idx >= config.log_interval:
                        break
                    c = c.to(device)
                    t = t.to(device)
                    logits_eval = model(c)
                    logits_flat = logits_eval.view(-1, logits_eval.size(-1))
                    targets_flat = t.view(-1)
                    loss_eval_value = loss(logits_flat, targets_flat)
                    total_eval += loss_eval_value.item()
                    count += 1

            eval_loss = total_eval / max(count, 1)
            
            ### ======== TODO : END ========= ###
            
            print(
                f"Iteration {i}, Train Loss: {train_loss}",
                f"Eval Loss: {eval_loss}",
            )

            if config.to_log:
                wandb.log(
                    {
                        "Train Loss": train_loss,
                        "Eval Loss": eval_loss,
                    }
                )

            model.train()

        # Save the model every config.save_iterations
        if i % config.save_iterations == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "iteration": i,
                },
                config.save_path / f"mini_model_checkpoint_{i}.pt",
            )

        if i > config.max_iter:
            break

    print("Training complete.")
    return