# an individual in GA as an array

const INVALID_FITNESS = -1.0
const INVALID_COMPLEXITY = -1.0

"""

Each individual forms a tree of three levels. 
The root node is called a disjunction node (DN). The child nodes of a DN are each called a conjunction node (CN).
The leaf node denotes a variable or its negation and is called a variable node (VN).
In this struct, each non-leaf node is defined as a vector that specifies the edges from it to nodes 
in the next level.
"""
struct Individual
    DN::Vector{Bool}        # 1: activate the CN, 0: deactivate the CN
    CNs::Matrix{Int8}       # each column is a CN, 1: x, -1:  not x, 0: unselected

    fitness::Float64        # ∈[0, 1]
    complexity::Float64     # ≥ 0
end


get_num_inputs(ind::Individual) = size(ind.CNs, 1)
get_num_conjunctions(ind::Individual) = size(ind.CNs, 2)

"""
    Individual(num_inputs::Int, num_conjunctions::Int) -> Individual

Create a random individual.
"""
function Individual(num_inputs::Int, num_conjunctions::Int)
    cps = rand(Int8[-1, 0, 1], num_inputs, num_conjunctions)
    dp = rand(Bool, num_conjunctions)
    return Individual(dp, cps, INVALID_FITNESS, INVALID_COMPLEXITY)
end

"""
    to_expression(ind::Individual, feature_names::AbstractVector{String}) -> String

Return the Boolean expression represented by `ind` as a string using the Julia syntax. 
E.g., "(A & B) | (A & ~C)". If no CN node is activated, then return "true".
"""
function to_expression(ind::Individual, feature_names::AbstractVector{String})
    @assert get_num_inputs(ind) == length(feature_names)
    neg_feature_names = ["~" * name for name in feature_names]
    cp_strings = String[]
    cp_string_vector = String[]
    for i in eachindex(ind.DN)
        if ind.DN[i]    # CN i is activated
            empty!(cp_string_vector)
            cp = @view ind.CNs[:, i]
            for j in eachindex(cp)
                if cp[j] == 1
                    push!(cp_string_vector, feature_names[j])
                elseif cp[j] == -1
                    push!(cp_string_vector, neg_feature_names[j])
                end
            end
            if !isempty(cp_string_vector)
                push!(cp_strings, join(cp_string_vector, " & "))
            end
        end  
    end
    if isempty(cp_strings)
        return "true"
    end
    if length(cp_strings) == 1
        return cp_strings[1]
    end
    cp_strings .= ["(" * cp_str * ")" for cp_str in cp_strings]     # add brackets
    return join(cp_strings, " | ")
end


"""
    select_regulators(ind::Individual, feature_names::AbstractVector{String}) -> Vector{String}

Get the list of regulators that have been chosen by `ind`.
"""
function select_regulators(ind::Individual, feature_names::AbstractVector{String})
    @assert get_num_inputs(ind) == length(feature_names)
    regulator_indices = Set{Int}()
    for i in eachindex(ind.DN)
        if ind.DN[i]    # cp i is activated
            cp = @view ind.CNs[:, i]
            for j in eachindex(cp)
                if cp[j] == 1 || cp[j] == -1
                    push!(regulator_indices, j)
                end
            end
        end
    end
    return [feature_names[i] for i in regulator_indices]
end

"""
    crossover!(ind1::Individual, ind2::Individual; cp_cx_rate=0.1, edge_cx_rate=0.05)

Crossover two individuals in place. Here, `CN_subtree_cx_rate` specifies the probability that
two subtrees rooted at a CN at the same position are exchanged; `edge_cx_rate` specifies the
probability that every edge is exchanged.
"""
function crossover!(ind1::Individual, ind2::Individual; CN_subtree_cx_rate=0.1, edge_cx_rate=0.05)
    for i in 1:get_num_conjunctions(ind1)
        if rand() < CN_subtree_cx_rate
            ind1.CNs[:, i], ind2.CNs[:, i] = @views ind2.CNs[:, i], ind1.CNs[:, i]
        end
    end
    # exchange edges
    for i in eachindex(ind1.DN)
        if rand() < edge_cx_rate
            ind1.DN[i], ind2.DN[i] = ind2.DN[i], ind1.DN[i]
        end
    end
    for i in eachindex(ind1.CNs)
        if rand() < edge_cx_rate
            ind1.CNs[i], ind2.CNs[i] = ind2.CNs[i], ind1.CNs[i]
        end
    end
end

"""
    mutate!(ind::Individual; mut_reduce_rate=0.1, mut_add_rate=0.1)

Mutate an individual `ind` in place. Here, `mut_reduce_rate` is the probability that an edge is removed,
and `mut_increase_rate` is the probability that an edge is added.
"""
function mutate!(ind::Individual; mut_reduce_rate=0.1, mut_add_rate=0.1)
    # DP edges 
    for i in eachindex(ind.dp)
        r = rand()
        if r < mut_reduce_rate  # reduce an edge
            ind.DN[i] = false
        elseif r < mut_reduce_rate + mut_add_rate  # add an edge 
            ind.CN[i] = true
        end
    end
    # CP edges
    for CN in eachcol(ind.CNs)
        for i in eachindex(CN)
            r = rand()
            if r < mut_reduce_rate
                CN[i] = 0
            elseif r < mut_reduce_rate + mut_add_rate
                CN[i] = rand((1, -1))
            end
        end
    end
end