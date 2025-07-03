# attorch implementation of BN:
# https://github.com/BobMcDear/attorch/blob/main/attorch/batch_norm_kernels.py
# where:
#   x.shape = [N, C, L]
#   block.shape = [ceil(N), 1, ceil(L)]; loop on L with chunk size BLOCK_L

# Problem: locality?

# My plan:
#   block.shape = [ceil(N), 1, BLOCK_L]; loop on N with chunk size 1
