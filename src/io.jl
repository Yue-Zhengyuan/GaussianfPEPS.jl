"""
Load a TensorKit tensor
"""
function load_tensor(filename::AbstractString)
    @assert endswith(filename, ".jld2")
    return convert(TensorMap, JLD2.load_object(filename))
end

"""
Save a TensorKit tensor 
"""
function save_tensor(filename::AbstractString, t::AbstractTensorMap)
    @assert endswith(filename, ".jld2")
    return JLD2.save_object(filename, convert(Dict, t))
end
