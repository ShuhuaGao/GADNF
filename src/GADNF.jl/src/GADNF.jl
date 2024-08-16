module GADNF

using Printf
using JLD2
using Random
using SymbolicUtils

const AMI = AbstractMatrix{<:Integer}
const AVI = AbstractVector{<:Integer}
const AVB = AbstractVector{Bool}

include("bool_utils.jl")
include("individual.jl")
include("fitness.jl")
include("selection.jl")
include("ga.jl")

export Individual, run_GA, GAConfig, to_expression, to_simplified_expression, generate_bool_data,
    get_fitness, run_CGA, compute_distance, less_lex

end # module GADNF
