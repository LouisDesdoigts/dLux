import jax.numpy as np
import jax 


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

    i: int = np.arange(1, n + 1, dtype=int)
    j: int = np.zeros(shape, dtype=int)
    
    j: int = j.at[0].set(np.repeat(i, i, total_repeat_length=tot_len))

    # Now comes the problem of the inner for loop. I cannot use `for` 
    # because the loop variable is not static. However, I need to 
    # apply `np.arange` to multiple different values. Essentially,
    # I am looking for `np.arange(np.arange())`. This is a folly 
    # bacause the size of the array is unkown.

    # OK I'm starting to formulate a plan. So if I have an `(n, tot_len)` 
    # array of ones, I can zero out below the diagonal and then sum 
    # along the vertical axis to get the values that I need. Haha, yes!
    # For example,
    #
    #     1 1 1 1 1 1 => 63 
    #     0 0 1 0 1 1 => 11
    #     0 0 0 0 0 1 =>  1
    # sum 1 1 2 1 2 3
    #
    # Can this be done with `lax.scan`? Or perhaps I can be very clever 
    # and create the encoding above from a binary image of `n`? 
    #
    # n: 0 => b0
    # n: 1 => b1
    # n: 2 => b10
    # n: 3 => b11
    # n: 4 => b100
    # 
    # So b11...1 can be created how? Well it is the sum over `2 >> i`.
    # I cannot remeber if the `operator` library employs the vectorisable
    # versions of bitwise operations. It does, nice!
    #
    # np.sum(jax.vmap(operator.lshift)(i))
    # np.sum(jax.vmap(lambda x: 2 << x)(i))
    # np.sum(jax.vmap(lambda x: 2 ** x)(i))
    #
    # I'll use the bottom one for now but the other two present optimisation 
    # opportunities. 

    # While I hvae made some progress above I do not think that it is 
    # going to lead to a readable solution soon. Let me write done the 
    # naive approach. 
    #

    len_of_arange: callable = lambda x: x * (x + 1) / 2
    vmap_len_of_arange: callable = jax.vmap(len_of_arange)

    # Let me work through some examples,
    #
    #              1 | 1 2 | 1 2 3 | 1 2 3 4 |
    # start_index: 0 | 1   | 3     | 6   
    # length:      1 | 2   | 3     | 4
    #
    # God this is a slog. Just think: if I can get it, the reward will 
    # be worth it. 
    for k in np.arange(1, n + 1):
       l: int = (k * (k + 1) / 2)
       i: int = i.at[(l):(l + k)].set(np.arange(l))
    # 

    return j

j: int = jit_nested_for()
print(j)
