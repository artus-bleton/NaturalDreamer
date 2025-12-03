import attridict
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, recurrent_size, latent_size, config, device):
        self.config   = config
        self.device   = device
        self.capacity = int(self.config.capacity)

        # Expériences brutes
        self.observations     = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.nextObservations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions          = np.empty((self.capacity, actions_size),       dtype=np.float32)
        self.rewards          = np.empty((self.capacity, 1),                  dtype=np.float32)
        self.dones            = np.empty((self.capacity, 1),                  dtype=np.float32)

        # Latents Dreamer (stochastique + déterministe)
        self.last_latents          = np.empty((self.capacity, latent_size),    dtype=np.float32)
        self.last_recurrentStates  = np.empty((self.capacity, recurrent_size), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward,
            nextObservation, last_latent_state, last_recurrent_state, done):
        """Ajoute une transition + (z_t, h_t) dans le buffer."""
        i = self.bufferIndex

        self.observations[i]     = observation
        self.actions[i]          = action
        self.rewards[i]          = reward
        self.nextObservations[i] = nextObservation
        self.dones[i]            = done

        self.last_latents[i]          = last_latent_state.cpu()
        self.last_recurrentStates[i]  = last_recurrent_state.cpu()

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        if self.bufferIndex == 0:
            self.full = True

    def sample(self, batchSize, sequenceSize):
        """
        Retourne des séquences (B, T, ...) pour entraîner le world model.
        ⚠ Ne gère pas encore explicitement les 'done' (comme l’implé d’origine).
        """

        buf_len = len(self)
        max_start = buf_len - sequenceSize + 1
        assert max_start > 0, "not enough data in the buffer to sample"

        # indices de départ: (B,1)
        start_idx = np.random.randint(0, max_start, size=batchSize).reshape(-1, 1)
        # offsets temporels: (1,T)
        seq_offsets = np.arange(sequenceSize).reshape(1, -1)

        # indices finaux: (B,T)
        if self.full:
            sampleIndex = (start_idx + seq_offsets) % self.capacity
        else:
            sampleIndex = start_idx + seq_offsets

        observations      = torch.as_tensor(self.observations[sampleIndex],     device=self.device).float()
        nextObservations  = torch.as_tensor(self.nextObservations[sampleIndex], device=self.device).float()
        actions           = torch.as_tensor(self.actions[sampleIndex],          device=self.device).float()
        rewards           = torch.as_tensor(self.rewards[sampleIndex],          device=self.device).float()
        dones             = torch.as_tensor(self.dones[sampleIndex],            device=self.device).float()
        last_latents           = torch.as_tensor(self.last_latents[sampleIndex],          device=self.device).float()
        last_recurrentStates   = torch.as_tensor(self.last_recurrentStates[sampleIndex],  device=self.device).float()

        sample = attridict.AttriDict({
            "observations"     : observations,      # (B,T, *obs_shape)
            "nextObservations" : nextObservations,  # (B,T, *obs_shape)
            "actions"          : actions,           # (B,T, action_dim)
            "rewards"          : rewards,           # (B,T,1)
            "dones"            : dones,             # (B,T,1)
            "latents"          : last_latents,           # (B,T, latentSize)
            "recurrentStates"  : last_recurrentStates,   # (B,T, recurrentSize)
        })
        return sample
