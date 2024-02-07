# Import ML library
import Flux

# Import library to handle dataframes
import DataFrames as DF

## =============================================================================

"""
    seq_onehotbatch(seq_algn::Vector{<:String})

Convert a vector of sequences into a one-hot encoded 3D array.

# Arguments
- `seq_algn::Vector{<:String}`: A vector of sequences, where each sequence is a
  string.

# Returns
- A 3D array where each sequence is one-hot encoded. The unique characters in
  the sequences are determined, and each sequence is split into characters.  The
  sequences are then one-hot encoded and concatenated into a 3D array.

# Example
```julia
seq_onehotbatch(["ACTG", "TGCA"])
```
"""
function seq_onehotbatch(seq_algn::Vector{<:String})
    # Fin unique characters in the sequences
    unique_chars = sort(unique(join(seq_algn)))

    # Split each sequence into characters
    seq_split = [collect(x) for x in seq_algn]

    # One-hot encode the sequences concatenated into a 3D array
    return reduce(
        (x, y) -> cat(x, y, dims=3),
        Flux.onehotbatch.(seq_split, Ref(unique_chars))
    )
end # function