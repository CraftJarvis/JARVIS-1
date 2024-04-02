import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch as th

from jarvis.steveI.steveI_lib.VPT.lib.tree_util import tree_map


def print_recursive_shape(prefix, obj, indent_level=0, add_indent: bool = False):
    """Prints the shape of a nested object."""
    indent = ' ' * indent_level
    if add_indent:
        prefix = indent + prefix

    if indent_level == 0:
        print()

    if isinstance(obj, dict):
        print(f'{prefix} -> dict({len(obj)})')
        for k, v in obj.items():
            print_recursive_shape(f'{prefix}.{k}', v, indent_level + 1)
    elif isinstance(obj, list):
        print(f'{prefix} -> list({len(obj)})')
        for i, v in enumerate(obj):
            print_recursive_shape(f'{prefix}[{i}]', v, indent_level + 1)
    elif isinstance(obj, tuple):
        print(f'{prefix} -> tuple({len(obj)})')
        for i, v in enumerate(obj):
            print_recursive_shape(f'{prefix}[{i}]', v, indent_level + 1)
    elif obj is None:
        print(f'{prefix} -> None')
    else:
        print(f'{prefix} = {obj.shape}')


def batch_recursive_objects(ls, check_shape: bool = False):
    """Batch a list of objects into one object so the lowest level arrays are batched
    and everything else has the same structure.

    All objects in the list must have the same structure. 
    Concat along the already existing batch dimension.

    Simple example:
    >>> a = np.random.rand(1, 1, 4)
    >>> b = np.random.rand(2, 2, 4)
    >>> c = [{'a': a, 'b': b}, {'a': a, 'b': b}]
    >>> print_recursive_shape('c', c)
    c -> list(2)
    c[0] -> dict(2)
    c[0].a = (1, 1, 4)
    c[0].b = (2, 2, 4)
    c[1] -> dict(2)
    c[1].a = (1, 1, 4)
    c[1].b = (2, 2, 4)

    >>> print_recursive_shape('batch_recursive_objects(c)', batch_recursive_objects(c))
    batch_recursive_objects(c) -> dict(2)
    batch_recursive_objects(c).a = (2, 1, 4)
    batch_recursive_objects(c).b = (4, 2, 4)

    Complicated example:
    >>> a = np.random.rand(1, 1, 4)
    >>> b = np.random.rand(2, 2, 4)
    >>> c = {'a': a, 'b': b, 't': (a, b), 'n': None}
    >>> d = {'a': a, 'b': b, 't': (a, b), 'n': None}
    >>> e = [c, d]
    >>> print_recursive_shape('e', e)
    e -> list(2)
    e[0] -> dict(4)
    e[0].a = (1, 1, 4)
    e[0].b = (2, 2, 4)
    e[0].t -> tuple(2)
    e[0].t[0] = (1, 1, 4)
    e[0].t[1] = (2, 2, 4)
    e[0].n -> None
    e[1] -> dict(4)
    e[1].a = (1, 1, 4)
    e[1].b = (2, 2, 4)
    e[1].t -> tuple(2)
    e[1].t[0] = (1, 1, 4)
    e[1].t[1] = (2, 2, 4)
    e[1].n -> None

    >>> print_recursive_shape('batch_recursive_objects(e)', batch_recursive_objects(e))
    batch_recursive_objects(e) -> dict(4)
    batch_recursive_objects(e).a = (2, 1, 4)
    batch_recursive_objects(e).b = (4, 2, 4)
    batch_recursive_objects(e).t -> tuple(2)
    batch_recursive_objects(e).t[0] = (2, 1, 4)
    batch_recursive_objects(e).t[1] = (4, 2, 4)
    batch_recursive_objects(e).n -> None
    """
    first = ls[0]
    if isinstance(first, dict):
        # Explanation for the below line:
        #   - ls[0] is a dict, so we can iterate over its keys
        #   - for each key, we get a list of values from each dict in ls
        #   - we batch the list of values
        #   - we return a dict with the same keys as ls[0] and the batched values
        return {k: batch_recursive_objects([d[k] for d in ls]) for k in first}
    elif isinstance(first, list):
        # Similar to the above, but for lists
        return [batch_recursive_objects([l[i] for l in ls]) for i in range(len(first))]
    elif isinstance(first, tuple):
        # Similar to the above, but for tuples
        return tuple(batch_recursive_objects([l[i] for l in ls]) for i in range(len(first)))
    elif first is None:
        return None
    else:
        if check_shape:  # might be slow
            assert all([e.shape == first.shape for e in ls]), \
                'All objects must have the same shape'

        if isinstance(first, np.ndarray):
            return np.concatenate(ls, axis=0)
        elif isinstance(first, th.Tensor):
            return th.cat(ls, dim=0)
        else:
            print(9)
            raise ValueError(f'Unsupported type: {type(first)}.'
                             'Only numpy arrays and torch tensors are supported '
                             'for non-(dict, list, tuple, None) objects')


def get_ith_slice_of_object(obj, i):
    """Given a batched object (output of batch_recursive_objects), 
    get the ith slice of the batch (with the same structure).

    IMPORTANT: 
    Only gets the the ith batch if each ith object had batch dim 1 before batching.
    batch_recursive_objects can technically batch objects with different batch dims,
    but that gets confusing and is not supported here. It also just doesnt make sense
    and should never happen in our case.

    Example:
    >>> a = np.random.rand(1, 1, 4)
    >>> b = np.random.rand(2, 2, 4)
    >>> c = {'a': a, 'b': b, 't': (a, b), 'n': None}
    >>> d = {'a': a, 'b': b, 't': (a, b), 'n': None}
    >>> e = [c, d]
    >>> batched = batch_recursive_objects(e)
    >>> print_recursive_shape('batched', batched)
    batched -> dict(4)
    batched.a = (2, 1, 4)
    batched.b = (4, 2, 4)
    batched.t -> tuple(2)
    batched.t[0] = (2, 1, 4)
    batched.t[1] = (4, 2, 4)
    batched.n -> None

    >>> print_recursive_shape('get_ith_batched_object(batched, 0)', get_ith_batched_object(batched, 0))
    get_ith_batched_object(batched, 0) -> dict(4)
    get_ith_batched_object(batched, 0).a = (1, 1, 4)
    get_ith_batched_object(batched, 0).b = (2, 2, 4)
    get_ith_batched_object(batched, 0).t -> tuple(2)
    get_ith_batched_object(batched, 0).t[0] = (1, 1, 4)
    get_ith_batched_object(batched, 0).t[1] = (2, 2, 4)
    get_ith_batched_object(batched, 0).n -> None

    """
    if isinstance(obj, dict):
        return {k: get_ith_slice_of_object(v, i) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [get_ith_slice_of_object(v, i) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(get_ith_slice_of_object(v, i) for v in obj)
    elif obj is None:
        return None
    else:
        if isinstance(obj, np.ndarray):
            return np.expand_dims(obj[i], axis=0)
        elif isinstance(obj, th.Tensor):
            return th.unsqueeze(obj[i], dim=0)
        else:
            raise ValueError(f'Unsupported type: {type(obj)}.'
                             'Only numpy arrays and torch tensors are supported '
                             'for non-(dict, list, tuple, None) objects')


def object_to_numpy(x):
    if isinstance(x, th.Tensor):
        return x.numpy()
    return x


def object_to_torch_and_device(x, device):
    def object_to_device(x):
        if isinstance(x, th.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return th.from_numpy(x).to(device)

    return tree_map(object_to_device, x)


@contextmanager
def timeit_context(label):
    t0 = time.time()
    yield
    print(f'\t{label} took {time.time() - t0:.2f} seconds')


class Timer:
    """Timer class for timing code blocks.
    Stores timings based on keys passed into the context manager.
    When Timer is turned into a dict, it returns a dict of the average timings
    per key. When reset is called, it resets the timings dict."""

    def __init__(self, name):
        self.timings = defaultdict(list)
        self.throughputs_n = defaultdict(int)
        self.start_time = time.time()
        self.name = name

    @contextmanager
    def time(self, key):
        t0 = time.time()
        yield
        self.timings[key].append(time.time() - t0)

    def reset(self):
        self.timings = defaultdict(list)
        self.throughputs_n = defaultdict(int)
        self.start_time = time.time()

    def dict(self):
        cur_time = time.time()
        timings = {f'{self.name}/{k}': np.mean(v) for k, v in self.timings.items()}
        # Add the throughputs
        for k, v in self.throughputs_n.items():
            timings[f'{self.name}/{k}'] = v / (cur_time - self.start_time)
        return timings

    def time_iter(self, iterable, key):
        """Time how long it takes to get the next item from an iterable."""
        if not hasattr(iterable, '__next__'):
            iterable = iter(iterable)
        while True:
            start_time = time.time()
            try:
                item = next(iterable)
            except StopIteration:
                break
            self.timings[key].append(time.time() - start_time)
            yield item

    def throughput(self, key, n):
        """This lets us record the throughput of various operations.
        For example, the number of tokens/second processed."""
        self.throughputs_n[key] += n
