import jax.numpy as np
import jax 

@jax.jit
def jit_nested_for():
    """
    The problem that this is trying to solve is.

    ```python
    n: int = 10
    for i in np.arange(n):
        for j in np.arange(i):
            pass
    ```
    """
    
    # So now the problem is that the generation of the ragged array.
    # perhaps I can do this by creating a function that "skips" zero 
    # values (i.e. using `jax.lax.cond`). Actually this could be 
    # a dectorator. This seems like a very complex solution to a 
    # simple problem to me. How about I make it flat? I.e. a long "set"
    # of tuples `(i, j)` pairs. This sounds very promising. Also I 
    # should write this stuff in my book instead of typing plans 
    # as comments. 

    # So I know the full size of the array and can work it out easily 
    # enough. Basically it is the sum of the integers less than `j`.
    # This can be calculated by multiplying the average with the 
    # number of terms. So this needs to be coded in such a way that
    # it is completely independent of `i`. Otherwise this will break. 
    # Well I am generating numbers from `1 -> n + 1` so this implies 
    # that the mean is going to be only a function of `n`.

    # Let me work through some examples,
    #
    # n: 2
    # i: 1, 2
    # j: 1, 1
    #       2
    # (i, j) => ((1, 1), (2, 1), (2, 2))
    # 
    # n: 3
    # i: 1, 2, 3
    # j: 1, 1, 1
    #       2, 2
    #          3
    # (i, j) => ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
    #
    # n | length
    # ----------
    # 2 | 3
    # 3 | 6
    # 
    # Ahhhh, the internet has provided. The answer that I was looking 
    # for originally was,
    #
    # S = n(n + 1) / 2
    #
    # Let me see if `n` is even then `n + 1` is odd and `n (n + 1)` is 
    # even so the division by two is always an integer. 
    # So this is similar to `itertools.starmap` and also reminds me of 
    # damn it ... I had it right on the tip of my tongue. I wonder if 
    # this can be formulated as a tensor product?

    n: int = 5
    tot_len: int = int(n * (n + 1) / 2)
    shape: tuple = (2, tot_len) 

    i: int = np.arange(n, dtype=int)
    j: int = np.zeros(shape, dtype=int)

    return j

j: int = jit_nested_for()
print(j)
