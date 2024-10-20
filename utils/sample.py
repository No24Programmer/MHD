import torch
import random


def random_sample(list_of_tensors, sample_rate):
    if sample_rate < 1:
        # Assuming all tensors have the same number of samples in the first dimension
        num_samples = list_of_tensors[0].size(0)

        sample_size = int(num_samples * sample_rate)

        # Generate random indices for sampling
        indices = random.sample(range(num_samples), sample_size)

        # Perform sampling on each tensor
        sampled_tensors = []
        for tensor in list_of_tensors:
            sampled_tensor = tensor[indices]
            sampled_tensors.append(sampled_tensor)

        return sampled_tensors
    else:
        return list_of_tensors
