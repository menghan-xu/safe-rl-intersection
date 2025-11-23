import yaml
from argparse import ArgumentParser
from collections import namedtuple
from ppo import train, make_env, val
import gymnasium as gym
import torch 
import numpy as np


def test_policy(env_id, path):
    
    with open(path, "r") as f:
        hyps = yaml.safe_load(f)
    model = train(env_id=env_id,**hyps[env_id])

    torch.save(model, f"learned_policies/{args.env_id}/model.pt")
    
    env = make_env(env_id)()
    # model = torch.load(f"learned_policies/{args.env_id}/model.pt")
    avg_rew = val(model, env, 100)
    assert avg_rew > {"CartPole-v0":195, "LunarLander-v2": 195, "Reacher-v4": -7}[env_id]
    print("Achieved reward", avg_rew)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--path", type=str, default="hyperparameters.yaml")
    args = parser.parse_args()
    test_policy(args.env_id, args.path)