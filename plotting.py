import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import smooth


def plot_loss(losses, ax=None, ylabel="Loss"):
    if ax is None:
        plt.plot(losses)
        plt.ylabel(ylabel)
        plt.xlabel("Updates")
        plt.show()
    else:
        ax.plot(losses)

def plot_reward(train_reward, eval_reward, eval_episodes, ax=None):
    if ax is None:
        plt.plot(smooth(train_reward, 5), label="train")
        # smooth_eval_rewards = smooth(eval_reward, 2)
        # plt.plot(eval_episodes[:len(smooth_eval_rewards)], smooth_eval_rewards, label="eval", marker=".")
        plt.plot(eval_episodes, eval_reward, label="eval", marker=".")
        plt.ylabel("Reward")
        plt.xlabel("Episodes")
        plt.legend()
        plt.show()
    else:
        ax.plot(smooth(train_reward, 5), label="train")
        ax.plot(eval_episodes, eval_reward, label="eval", marker=".")
        ax.legend()
