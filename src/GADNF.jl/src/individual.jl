# an individual in GA as an array

const INVALID_FITNESS = -1.0
const INVALID_COMPLEXITY = -1.0

"""
    mutable struct Individual

Each individual forms a tree of three levels. 
The root node is called a disjunction node (DN). The child nodes of a DN are each called a conjunction node (CN).
The leaf node denotes a variable or its negation and is called a variable node (VN).
In this struct, each non-leaf node is defined as a vector that specifies the edges from it to nodes 
in the next level.
"""
mutable struct Individual
    DN::Vector{Bool}        # 1: activate the CN, 0: deactivate the CN
    CNs::Matrix{Int8}       # each column is a CN, 1: x, -1:  not x, 0: unselected

    fitting_error_rate::Float64        # ∈[0, 1]
    complexity::Float64     # ≥ 0

    # the fields below are used as preallocated buffer to reduce memory allocations
    ŷ::BitVector                    # the prediction result
    regulator_selected::BitVector   # whether a regulator is selected
end


get_num_inputs(ind::Individual) = size(ind.CNs, 1)
get_num_conjunctions(ind::Individual) = size(ind.CNs, 2)

"""
    vec(ind::Individual) -> Vector

Linear representation of the individual `ind` as a vector.
Form: `[ind.DN; vec(ind.CNs)]`.
"""
function Base.vec(ind::Individual)
    data = [ind.DN; vec(ind.CNs)]
    return data
end

"""
    Individual(num_inputs::Int, num_conjunctions::Int) -> Individual

Create a random individual.`max_samples` indicates the maximum number of samples during training, 
whose default value is 1000. This parameter is used to preallocate memory for speedup.
"""
function Individual(num_inputs::Int, num_conjunctions::Int)
    cps = rand(Int8[-1, 0, 1], num_inputs, num_conjunctions)
    dp = rand(Bool, num_conjunctions)
    ŷ = BitVector()
    regulator_selected = zeros(Bool, num_inputs)
    return Individual(dp, cps, INVALID_FITNESS, INVALID_COMPLEXITY, ŷ, regulator_selected)
end

is_evaluated(ind::Individual) = ind.fitting_error_rate != INVALID_FITNESS && ind.complexity != INVALID_COMPLEXITY

"""
    to_expression(ind::Individual, feature_names::AbstractVector{String}) -> String

Return the Boolean expression represented by `ind` as a string using the Julia syntax. 
E.g., "(A & B) | (A & ~C)". If no CN node is activated, then return "false".
"""
function to_expression(ind::Individual, feature_names::AbstractVector{String}=String[])
    if isempty(feature_names)
        ni = get_num_inputs(ind)
        feature_names = ["x$i" for i in 1:ni]
    else
        @assert get_num_inputs(ind) == length(feature_names)
    end
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
        return "false"
    end
    if length(cp_strings) == 1
        return cp_strings[1]
    end
    cp_strings .= ["(" * cp_str * ")" for cp_str in cp_strings]     # add brackets
    return join(cp_strings, " | ")
end

"""
    to_simplified_expression(ind::Individual, feature_names::AbstractVector{String}) -> SymbolicUtils.BasicSymbolic{Bool}

Return the simplified Boolean expression represented by `ind` as a symbolic expression.
If no node is activated effectively, then return `false`.
"""
function to_simplified_expression(ind::Individual, feature_names::AbstractVector{String}=String[])
    if isempty(feature_names)
        ni = get_num_inputs(ind)
        feature_names = ["x$i" for i in 1:ni]
    else
        @assert get_num_inputs(ind) == length(feature_names)
    end
    # create symbolic variables
    svs = [SymbolicUtils.Sym{Bool}(Symbol(v)) for v in feature_names]
    cts = SymbolicUtils.BasicSymbolic{Bool}[]    # all activated conjunction terms
    ct = SymbolicUtils.BasicSymbolic{Bool}[]    # one conjunction term 
    for i in eachindex(ind.DN)
        if ind.DN[i]    # CN i is activated
            cp = @view ind.CNs[:, i]
            empty!(ct)
            for j in eachindex(cp)
                if cp[j] == 1
                    push!(ct, svs[j])
                elseif cp[j] == -1
                    push!(ct, ~svs[j])
                end
            end
            if !isempty(ct)
                push!(cts, reduce(&, ct))
            end
        end
    end
    if isempty(cts)
        return false
    end 
    # do disjunction and simplify
    expr = reduce(|, cts)
    return simplify_bool(expr)
end

"""
    mark_selected_regulators!(ind::Individual) -> BitVector

Mark the selected regulators in `ind` in place. The resultant selector vector is also returned.
"""
function mark_selected_regulators!(ind::Individual)::BitVector
    fill!(ind.regulator_selected, false)
    for i in eachindex(ind.DN)
        if ind.DN[i]    # cp i is activated
            cp = @view ind.CNs[:, i]
            for j in eachindex(cp)
                if cp[j] == 1 || cp[j] == -1   # C-edge j is activated
                    ind.regulator_selected[j] = true
                end
            end
        end
    end
    return ind.regulator_selected
end

"""
    get_selected_regulators(regulator_selected::AVB, feature_names::AbstractVector{String}) -> Vector{String}

Given a selector `regulator_selected`, return the names of the selected regulators.
"""
function get_selected_regulators(regulator_selected::AVB, feature_names::AbstractVector{String})
    return feature_names[regulator_selected]
end

"""
    crossover!(ind1::Individual, ind2::Individual; cp_cx_rate=0.1, edge_cx_rate=0.05)

Crossover two individuals in place. Here, `CN_subtree_cx_rate` specifies the probability that
two subtrees rooted at a CN at the same position are exchanged; `edge_cx_rate` specifies the
probability that an edge is exchanged.
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
    nothing
end

"""
    mutate!(ind::Individual; mut_reduce_rate=0.1, mut_add_rate=0.1)

Mutate an individual `ind` in place. Here, `mut_reduce_rate` is the probability that an edge is removed,
and `mut_increase_rate` is the probability that an edge is added.
"""
function mutate!(ind::Individual; mut_reduce_rate=0.1, mut_add_rate=0.1)
    # DN edges 
    for i in eachindex(ind.DN)
        r = rand()
        if r < mut_reduce_rate  # reduce an edge
            ind.DN[i] = false
        # elseif r < mut_reduce_rate + mut_add_rate  # add an edge '
        
            ind.DN[i] = true
        end
    end
    # CN edges
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
    nothing
end


function mutate2!(ind::Individual; mut_reduce_rate=0.1, mut_rate=0.1)
    if rand() < mut_rate
        if rand() < 0.5     # mutate DN edges
            i = rand(1:get_num_conjunctions(ind))
            ind.DN[i] = !ind.DN[i]
        else                # mutate CN edges
            # select a CN edge randomly
            i = rand(1:get_num_conjunctions(ind))
            j = rand(1:get_num_inputs(ind))
            if rand() < mut_reduce_rate
                ind.CNs[j, i] = 0   # remove a CN edge
            else
                ind.CNs[j, i] = rand((1, -1))   # add a CN edge
            end
        end
    end
    nothing
end


function Base.display(ind::Individual)
    display(ind.DN)
    display(ind.CNs)
end