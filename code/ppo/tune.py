from argparse import ArgumentParser
from ppo import train, val, make_env
import numpy as np
import torch


def sample_params(env_id):
    # For reference, you may not want to tune all the parameters
    epochs = np.random.choice([100, 250, 350, 500])
    gamma = np.random.uniform(0.95, 0.999)
    gae_lambda = np.random.uniform(0.9, 1)
    lr = np.random.uniform(3e-6, 3e-4)
    num_steps = np.random.choice([128, 256, 512, 1024])
    minibatch_size = np.random.choice([16, 32, 64])
    clip_ratio = np.random.choice([0.1, 0.2, 0.3, 0.5])
    num_envs = np.random.choice([2, 4, 8, 16])
    ent_coef = np.random.choice([0, 1e-2, 1e-3, 1e-4])
    vf_coef = np.random.choice([0.25, 0.5, 1])
    seed = 42
    return dict(
        epochs=epochs, 
        gamma=gamma,
        gae_lambda=gae_lambda,
        lr=lr,
        num_steps=num_steps,
        minibatch_size=minibatch_size,
        clip_ratio=clip_ratio,
        num_envs=num_envs,
        ent_coef=ent_coef,
        seed=seed,
        vf_coef=vf_coef
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--num_trials", type=int, default=100)
    args = parser.parse_args()
    for i in range(args.num_trials):
        train_args = sample_params(args.env_id)
        model = train(args.env_id, **train_args)
        env = make_env(args.env_id)()
        eval_rew = val(model, env, num_ep=100)
        print(eval_rew, train_args)
