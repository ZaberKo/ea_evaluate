import numpy as np
from ray.rllib.utils.typing import ModelWeights


def mutate_inplace(gene: ModelWeights,
                   weight_magnitude: float = 1e7,
                   mut_strength=0.1,
                   num_mutation_frac=0.5):
    super_mut_strength = 10
    super_mut_prob = 0.05
    reset_prob = super_mut_prob + 0.02

    # num_params = len(gene)
    # ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

    mutate_count = 0
    for name, W in gene.items():  # Mutate each param
        if "_convs" in name or "_value" in name:
            continue

        need_reshape = False

        _W = np.squeeze(W)  # return a view of W (shadow copy)
        if len(_W.shape) <= 2 and len(W.shape) > 2:
            shape = W.shape
            need_reshape = True
            # W_ref=W
            W = _W

        if len(W.shape) == 2:  # Weights, no bias
            mutate_count += 1

            num_weights = W.shape[0] * W.shape[1]
            # Number of mutation instances
            num_mutations = np.ceil(
                num_mutation_frac * num_weights).astype(np.int)
            for _ in range(num_mutations):
                ind_dim1 = np.random.randint(0, W.shape[0])
                ind_dim2 = np.random.randint(0, W.shape[-1])
                random_num = np.random.rand()

                if random_num < super_mut_prob:  # Super Mutation probability
                    W[ind_dim1, ind_dim2] += np.random.normal(
                        0, super_mut_strength * np.abs(W[ind_dim1, ind_dim2]))
                elif random_num < reset_prob:  # Reset probability
                    W[ind_dim1, ind_dim2] = np.random.normal(0, 0.1)
                else:  # mutauion even normal
                    W[ind_dim1, ind_dim2] += np.random.normal(
                        0, mut_strength * np.abs(W[ind_dim1, ind_dim2]))

                # Regularization hard limit
                W[ind_dim1, ind_dim2] = np.clip(
                    W[ind_dim1, ind_dim2], -weight_magnitude, weight_magnitude)

        elif len(W.shape) == 1:  # Bias or layernorm
            mutate_count += 1
            num_weights = W.shape[0]

            # Number of mutation instances
            num_mutations = np.ceil(
                num_mutation_frac * num_weights).astype(np.int)
            for _ in range(num_mutations):
                ind_dim = np.random.randint(0, W.shape[0])
                random_num = np.random.rand()

                if random_num < super_mut_prob:  # Super Mutation probability
                    W[ind_dim] += np.random.normal(0,
                                                   super_mut_strength * np.abs(W[ind_dim]))
                elif random_num < reset_prob:  # Reset probability
                    W[ind_dim] = np.random.normal(0, 1)
                else:  # mutauion even normal
                    W[ind_dim] += np.random.normal(0,
                                                   mut_strength * np.abs(W[ind_dim]))

                # Regularization hard limit
                W[ind_dim] = np.clip(
                    W[ind_dim], -weight_magnitude, weight_magnitude)

        # if need_reshape:
        #     np.copyto(dst=W_ref,src=W.reshape(shape))

    print(f"mutated number of params: {mutate_count}")
