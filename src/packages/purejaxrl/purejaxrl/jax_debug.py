import jax
import jax.numpy as jnp
import os

from dataclasses import dataclass
from typing import Tuple, Any, Sequence
import collections
from jax import tree_util  # Import tree_util


def debuggable_conditional_breakpoint(pred):
    # see https://github.com/jax-ml/jax/issues/15039 for why this is the best way

    if os.environ.get("JAX_DISABLE_JIT", "").lower() == "true":
        jax.lax.cond(pred, lambda: breakpoint(), lambda *args: None)


def debuggable_pmap(fun, axis_name):
    if os.environ.get("JAX_DISABLE_JIT", "").lower() == "true":
        return loop_based_vmap_replacement(func, in_axes=in_axes, out_axes=out_axes)
    else:
        return jax.pmap(func, in_axes=in_axes, out_axes=out_axes)
            
def debuggable_vmap(func, in_axes=0, out_axes=0):
    """
    Conditionally returns either jax.vmap or loop_based_vmap_replacement
    based on the JAX_DISABLE_JIT environment variable.

    Args:
        func: The function to be vectorized.
        in_axes: Specifies the input axes to map over (passed to jax.vmap or replacement).
        out_axes: Specifies the output axes (passed to jax.vmap or replacement).

    Returns:
        Either jax.vmap or loop_based_vmap_replacement, depending on JAX_DISABLE_JIT.
    """
    if os.environ.get("JAX_DISABLE_JIT", "").lower() == "true":
        return loop_based_vmap_replacement(func, in_axes=in_axes, out_axes=out_axes)
    else:
        return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)


def debuggable_scan(body_fun, init, xs=None, length=None, reverse=False, unroll=1):
    """
    Conditionally returns either jax.lax.scan or loop_based_scan_replacement
    based on the JAX_DISABLE_JIT environment variable.

    Args:
        body_fun: The function to be scanned (passed to jax.lax.scan or replacement).
        init: The initial carry value (passed to jax.lax.scan or replacement).
        xs: (Optional) Input sequence (passed to jax.lax.scan or replacement).
        length: (Optional) Length of scan, if input sequence is not provided (passed to replacement).
        reverse: (Optional, bool) Reverse scan (passed to jax.lax.scan or replacement).
        unroll: (Optional, int or bool) Unrolling factor (passed to jax.lax.scan or replacement).

    Returns:
        Either jax.lax.scan or loop_based_scan_replacement, depending on JAX_DISABLE_JIT.
    """
    if os.environ.get("JAX_DISABLE_JIT", "").lower() == "true":
        return loop_based_scan_replacement(body_fun, init, xs=xs, length=length, reverse=reverse, unroll=unroll)
    else:
        return jax.lax.scan(body_fun, init, xs=xs, length=length, reverse=reverse, unroll=unroll)

def debuggable_unpack(x):
    if os.environ.get("JAX_DISABLE_JIT", "").lower() == "true":
        return x.unpack() if isinstance(x, DummyTracedArray) else x # Unpack here
    else:
        return x


class DummyTracedArray:
    """
    A dummy class to simulate a JAX traced array for debugging purposes.
    Provides ndim, shape, and basic slicing.  Improved for nested vmap and list data.
    """
    def __init__(self, data, shape: Tuple[int, ...], dtype):
        self._data = data # Store the actual data (could be anything - now handles lists robustly)
        self._shape = shape
        self._dtype = dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype
    
    def unpack(self):
        """Returns the underlying jnp.ndarray data."""
        return self._data
    
    def __repr__(self):
        # Enhanced repr to be more informative, like jax arrays
        dtype_str = str(self.dtype) if not isinstance(self.dtype, tuple) else ', '.join(map(str, self.dtype)) # Handle tuple dtypes
        data_preview = str(self._data)[:50] + "..." if isinstance(self._data, str) and len(str(self._data)) > 50 else str(self._data)[:50] #preview of data

        return f"DummyTracedArray(shape={self.shape}, dtype={dtype_str}, data={data_preview})"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index):
        # Enhanced __getitem__ to handle tuple indices and list data

        if isinstance(index, tuple):
            current_data = self._data
            current_shape_list = list(self.shape) # Make shape mutable for tracking

            indexed_data = current_data
            consumed_dims = 0

            for idx in index:
                if isinstance(indexed_data, list): # Handle list data explicitly for tuple indices
                    if isinstance(idx, int):
                        indexed_data = indexed_data[idx]
                        if current_shape_list: # Decrement shape dimension if possible for integers
                            current_shape_list.pop(0) # Consume a dimension with int index
                        consumed_dims += 1
                    elif isinstance(idx, slice):
                        indexed_data = indexed_data[idx] # Apply slice to list
                        if current_shape_list:
                            pass # Slice preserves dimension count, just changes size - shape will be updated in _infer_shape
                        consumed_dims += 1 # Consider slice as consuming a dimension in shape inference in terms of original dims
                    elif idx is Ellipsis:
                         break # Ellipsis - stop consuming dims after this (simplified)
                    else:
                        raise NotImplementedError(f"List indexing with type {type(idx)} in tuple index not yet supported")
                else: # Fallback to original indexing for non-list data (e.g., ndarrays within DummyTracedArray)
                    indexed_data = indexed_data[idx] # Use original indexing for non-list data

            if isinstance(indexed_data, DummyTracedArray): # If we get back another DummyTracedArray from slicing
                return indexed_data # Return it directly
            elif isinstance(indexed_data, list) or isinstance(indexed_data, jnp.ndarray):
                # Infer shape after indexing and return new DummyTracedArray
                inferred_shape = self._infer_shape_after_indexing(index) # Re-infer shape based on full index
                return DummyTracedArray(indexed_data, shape=tuple(inferred_shape), dtype=self.dtype) # Wrap new DummyTracedArray

            return indexed_data # If scalar or basic type, return directly

        elif isinstance(index, (int, slice)): # Handle single int or slice indices - mostly unchanged
            if isinstance(self._data, list):
                indexed_data = self._data[index] # Apply directly to list
            else:
                indexed_data = self._data[index]

            if isinstance(indexed_data, DummyTracedArray): # Propagate DummyTracedArray if we get one back
                return indexed_data
            elif isinstance(indexed_data, list) or isinstance(indexed_data, jnp.ndarray):
                inferred_shape = self._infer_shape_after_indexing((index,)) # Re-infer shape based on index
                return DummyTracedArray(indexed_data, shape=tuple(inferred_shape), dtype=self.dtype) # Wrap new DummyTracedArray
            return indexed_data # Scalar or basic type

        else:
            raise NotImplementedError(f"Indexing type {type(index)} not supported for DummyTracedArray yet")


    def _infer_shape_after_indexing(self, index_tuple: Tuple[Any, ...]) -> Tuple[int, ...]:
        # Shape inference - simplified and adjusted for list handling, more robust for mixed indexing
        current_shape = list(self._shape)
        new_shape = []
        index_dims_consumed = 0 # Track dims consumed

        for index in index_tuple:
            if not current_shape: # No more dimensions to index into
                break

            if isinstance(index, int):
                index_dims_consumed += 1 # Integer index removes a dimension
            elif isinstance(index, slice):
                start, stop, step = index.indices(current_shape[0])
                slice_len = max(0, (stop - start + step - 1) // step) # Correctly calculate slice length, handle negative steps if needed in future
                new_shape.append(slice_len)
                index_dims_consumed += 1
            elif index is Ellipsis: # ... Ellipsis - consume remaining dims (simplified)
                break # Stop dimension processing after ellipsis - simplified for now
            # For None index (new axis insertion) - simplified: skip for now for dummy array, can add if needed

            if index_dims_consumed <= len(current_shape): # Prevent index out of range on shape list
                if index_dims_consumed > 0: # Only remove from shape if a dimension was actually consumed by index type
                    current_shape = current_shape[index_dims_consumed:] # Update shape list by removing consumed dims
                    index_dims_consumed = 0 # Reset for next index in tuple


        return tuple(new_shape + current_shape) # Combine new dims from slices with remaining original dims

def maybe_wrap_results_in_stack(results, batch_size, out_axes):
    if not results:
        return jnp.array([])

    # --- Pytree based stacking ---
    first_result = results[0]
    flattened_first_result, treedef = tree_util.tree_flatten(first_result) # Flatten the first result to get treedef

    batched_leaves = []
    for leaf_index in range(len(flattened_first_result)):
        leaves_to_stack = [tree_util.tree_flatten(res)[0][leaf_index] for res in results] # Collect corresponding leaves
        stacked_leaf = jnp.stack(leaves_to_stack, axis=out_axes) # Stack leaves
        batched_leaves.append(stacked_leaf)
        #batched_leaves.append(DummyTracedArray(stacked_leaf, shape=stacked_leaf.shape, dtype=stacked_leaf.dtype)) # Wrap in DummyTracedArray

    return tree_util.tree_unflatten(treedef, batched_leaves) # Reconstruct using treedef and batched leaves



    # # Handle stacking of results - now handles tuple and custom object returns, wraps in DummyTracedArray
    # if not results:  # Handle empty results
    #     return jnp.array([]) # Or appropriate empty structure
    # elif isinstance(results[0], tuple):
    #     # If the function returns tuples, stack each element of the tuples separately
    #     num_tuple_elements = len(results[0])
    #     stacked_results = []
    #     for element_index in range(num_tuple_elements):
    #         elements_to_stack = [res[element_index] for res in results]
    #         # Wrap stacked tuple elements in DummyTracedArray if needed, otherwise keep as list
    #         if isinstance(elements_to_stack[0], jnp.ndarray) or jnp.isscalar(elements_to_stack[0]):
    #             stacked_results.append(jnp.stack(elements_to_stack, axis=out_axes))
    #         else: # Assume custom objects, keep as list of objects in DummyTracedArray
    #             stacked_results.append(DummyTracedArray(elements_to_stack, shape=(batch_size,), dtype=object)) # dtype=object for custom classes - adjust dtype if known
    #     return tuple(stacked_results)
    # elif isinstance(results[0], jnp.ndarray) or jnp.isscalar(results[0]):
    #     # If the function returns single arrays or scalars, stack and wrap in DummyTracedArray
    #     stacked_array = jnp.stack(results, axis=out_axes)
    #     return DummyTracedArray(stacked_array, shape=stacked_array.shape, dtype=stacked_array.dtype)
    # else:
    #     # If the function returns custom objects, wrap the list of objects in DummyTracedArray
    #     return DummyTracedArray(results, shape=(batch_size,), dtype=object) # dtype=object for custom classes - adjust dtype if known
        
def loop_based_vmap_replacement(func, in_axes=0, out_axes=0):
    """
    Returns a function that mimics jax.vmap but uses a Python for loop,
    handling in_axes with None values.

    This is intended as a *debugging* aid, not for performance. It will be
    significantly slower than jax.vmap.

    Args:
        func: The function to be vectorized.
        in_axes:  Specifies the input axes to map over.  Can be an integer or a tuple
                   of integers, or a tuple containing None values.
                   Defaults to 0.
        out_axes: Specifies the output axes to place mapped axes. Can be an integer
                   or a tuple of integers. Defaults to 0.

    Returns:
        A new function that, when called with arguments, will perform the
        vectorized operation using a Python for loop.
    """

    def vectorized_func_loop(*args):
        # Handle in_axes and out_axes
        if isinstance(in_axes, int):
            in_axes_tuple = (in_axes,) * len(args)
        else:
            in_axes_tuple = in_axes

        # Determine the length of the mapped dimension (batch size) - find the first non-None in_axis
        batch_size = None
        for arg_index, axis in enumerate(in_axes_tuple):
            if axis is not None:
                batch_size = args[arg_index].shape[axis]
                break
        if batch_size is None:
            # If all in_axes are None, then the batch size doesn't really matter for looping,
            # but to proceed we'll assume batch size of 1 (or take size from first arg if available)
            batch_size = 1
            if args:
                batch_size = args[0].shape[0] if args[0].ndim > 0 else 1 # Fallback, might need more robust logic

        results = []
        for i in range(batch_size):
            # Prepare arguments for the original function for this iteration
            single_element_args = []
            for arg_index, arg in enumerate(args):
                axis_to_slice = in_axes_tuple[arg_index]

                if axis_to_slice is None:
                    single_element_args.append(arg)
                else:
                    def slicing_fn(leaf): 
                        if isinstance(leaf, jnp.ndarray):
                            slices = [slice(None)] * leaf.ndim
                            slices[axis_to_slice] = i
                            return leaf[tuple(slices)]
                        else:
                            return leaf

                    sliced_arg = tree_util.tree_map(slicing_fn, arg) # Apply slicing_fn to all leaves of 'arg'
                    single_element_args.append(sliced_arg)


                    # sliced_leaves = []
                    # flattened_arg, treedef = tree_util.tree_flatten(arg)

                    # for leaf_index in range(len(flattened_arg)):
                    #     slices = [slice(None)] * flattened_arg[leaf_index].ndim
                    #     slices[axis_to_slice] = i
                    #     sliced_leaves.append(flattened_arg[leaf_index][tuple(slices)])

                    # return tree_util.tree_unflatten(treedef, sliced_leaves)


            # Call the original function with single elements
            result = func(*single_element_args)
            results.append(result)
        return maybe_wrap_results_in_stack(results, batch_size, out_axes)

    return vectorized_func_loop


def loop_based_scan_replacement(body_fun, init, xs=None, length=None, reverse=False, unroll=1):
    """
    Returns the result of mimicking jax.lax.scan using a Python for loop.
    Now directly returns (carry, output_sequence), matching jax.lax.scan's return.

    This is intended as a *debugging* aid, not for performance. It will be
    significantly slower than jax.lax.scan.

    Args:
        body_fun: The function to be scanned, with signature (carry, x) -> (new_carry, y).
        init: The initial carry value.
        xs: (Optional) The input sequence.
        length: (Optional) The length of the scan, if the input is not provided as xs.
        reverse: (Optional, bool) Whether to perform the scan in reverse. Defaults to False.
        unroll: (Optional, int or bool) Unrolling factor.  Basic handling for int > 1 and True (=1).
                Defaults to 1.  Boolean True treated as unroll=1 for simplicity.

    Returns:
        The final (carry, output_sequence) tuple, mimicking jax.lax.scan's return.
    """

    carry = init
    output_sequence = []
    input_sequence = xs

    if length is None:
        if input_sequence is None:
            raise ValueError("Input sequence 'xs' must be provided if length is not specified.")
        scan_length = input_sequence.shape[0] if jnp.ndim(input_sequence) > 0 else 1 # Handle scalar case
    else:
        scan_length = length

    indices = range(scan_length)
    if reverse:
        indices = reversed(indices)

    for _ in range(unroll): # Basic unrolling (for debugging, not full performance unrolling)
        for i in indices:
            x = input_sequence[i] if input_sequence is not None else None # Handle case with no input sequence
            carry, y = body_fun(carry, x) # Pass args and kwargs if needed (currently not passed for simplicity, add if needed)
            output_sequence.append(y)

    # Handle stacking of output sequence -

    stack = maybe_wrap_results_in_stack(output_sequence, scan_length, out_axes=0)
    return carry, stack

 


