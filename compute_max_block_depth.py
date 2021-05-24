def compute_max_block_depth(shape: int = 1920, print_out: bool = True):
    """
    Function which computes the max n_blocks possible to prevent maxpooling on
    odd spatial dimensions. Otherwise one risks semantic mismatches. If enabled,
    prints out a list of each level's shape.

    param: shape: int       Shape of input
    param: print_out: bool  Print out the result
    return: max_n_block     Max level of depth
    """

    if print_out:
        print(f'Input size: {shape}\n')

    level = 0
    while shape % 2 == 0 and shape / 2 > 1:
        level += 1
        shape /= 2

        if print_out:
            print(f'Level {level}: {shape}')

    if print_out:
        print(f'Max level: {level}')

    return level


if __name__ == '__main__':
    shape = 512
    n_block = compute_max_block_depth(shape)

