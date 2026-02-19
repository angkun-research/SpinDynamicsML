import h5py
import numpy as np

def _resolve_h5obj(obj, h5file):
    """Recursively resolve a Group/Dataset that may contain object references used by JLD2.

    - If `obj` is a Group, return a dict of resolved members.
    - If `obj` is a Dataset of numeric types, return the numpy array.
    - If `obj` is a Dataset of object/reference dtype, follow references and return
      an array (or nested list) of resolved arrays/scalars.
    """
    # Group: map children
    if isinstance(obj, h5py.Group):
        return {k: _resolve_h5obj(obj[k], h5file) for k in obj.keys()}

    # Dataset: try to read
    if isinstance(obj, h5py.Dataset):
        data = obj[...]

        # If data contains HDF5 references or dtype object, follow references
        if data.dtype == object or data.dtype.kind == 'O':
            # Flatten, resolve each reference if possible, then reshape
            flat = data.flatten()
            out = []
            for el in flat:
                # el is often an HDF5 object reference (h5py.Reference)
                try:
                    if isinstance(el, h5py.Reference) and el:  # valid reference
                        resolved = _resolve_h5obj(h5file[el], h5file)
                    else:
                        # If it's not a reference, it may already be a scalar/bytes
                        resolved = el
                except Exception:
                    # Fallback: return the element as-is
                    resolved = el
                out.append(resolved)
            # Try to convert list to numpy array where possible, otherwise keep list structure
            try:
                arr = np.array(out, dtype=object).reshape(data.shape)
            except Exception:
                arr = np.array(out, dtype=object)
            return arr

        # Otherwise return numeric data as numpy array
        return data

    # Fallback: return as-is
    return obj


def read_jld2_variables(fname, var_names=None):
    """Read the provided var names from a JLD2 file and resolve nested object refs.

    Returns a dict mapping var_name -> resolved numpy array / dict structure.
    """
    out = {}
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
        #print("Keys found in file:", keys)
        if var_names is None:
            var_names = keys

        for var in var_names:
            if var in f:
                out[var] = _resolve_h5obj(f[var], f)
            else:
                out[var] = None
    return out


if __name__ == '__main__':
    taskid = 15
    num_tasks = 250
    fname = f"data/HaoResults/results_task{taskid}_of_{num_tasks}.jld2"

    var_names = ['Js_list', 'as_list', 'bs_list', 'norm2s_list', 'B_list']

    data = read_jld2_variables(fname, var_names)

    # Print concise summary for each variable
    for var, val in data.items():
        if val is None:
            print(f"{var} not found in file.")
            continue

        # If it's an array of objects (common for lists of arrays), inspect first few entries
        if isinstance(val, np.ndarray) and val.dtype == object:
            print(f"{var}: object array, shape={val.shape}")
            # show the first element
            print(f"First element: {val.flat[0]}")
        else:
            try:
                print(f"{var}: type={type(val)}, shape={np.shape(val)}")
                print(f"First entries: {val.flat[0]}")
            except Exception:
                print(f"{var}: type={type(val)}")