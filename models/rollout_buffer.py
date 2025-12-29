import numpy as np
import random
import torch
from multiprocessing import Lock

class Rollout_Buffer:
    def __init__(self, manager=None):
        self.lock = manager.Lock() if manager else Lock()
        self.memory = manager.list() if manager else []

    def remember(self, transition):
        """
            存储单步数据 (s, a, adv_val, ns, d, lp)
        """
        with self.lock:
            self.memory.append(transition)

    def extend(self, batch_data):
        """
            存储一批 (s, a, adv_val, ns, d, lp)
        """
        with self.lock:
            self.memory.extend(batch_data)

    def get_all_and_clear(self):
        with self.lock:
            batch = list(self.memory)
            del self.memory[:]
        return self._process_batch(batch)

    def _process_batch(self, batch):
        if not batch: return None, None, None, None, None, None
        states, actions, rewards, next_states, dones, log_probs = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        if log_probs[0] is not None:
            log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32)
        else:
            log_probs = None

        return states, actions, rewards, next_states, dones, log_probs

    def clear(self):
        with self.lock:
            del self.memory[:]

    def __len__(self):
        return len(self.memory)