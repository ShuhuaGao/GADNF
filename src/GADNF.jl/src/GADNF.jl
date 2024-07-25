module GADNF

const AMI = AbstractMatrix{<:Integer}
const AVI = AbstractVector{<:Integer}
const AVB = AbstractVector{Bool}

include("individual.jl")
include("fitness.jl")
include("ga.jl")

end # module GADNF
