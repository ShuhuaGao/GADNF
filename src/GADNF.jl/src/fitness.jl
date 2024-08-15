# fitness metric 

"""
    evaluate(ind::Individual, x::BitVector) -> Bool

Evaluate `ind` on a sample `x`.
"""
function evaluate(ind::Individual, x::AbstractVector{Bool})::Bool
    @assert get_num_inputs(ind) == length(x)
    result = false
    for i in eachindex(ind.DN)
        if ind.DN[i]        # C-node i is activated
            CN = @view ind.CNs[:, i]
            if iszero(CN)    # all 0:no variable is chosen at this node
                continue
            end
            CN_result = true
            for i in eachindex(CN)
                if CN[i] == 1
                    CN_result &= x[i]
                elseif CN[i] == -1
                    CN_result &= ~x[i]
                end
            end
            result |= CN_result
            if result       # the final result must be true; no need to check more
                return true
            end
        end
    end
    return result
end


evaluate(ind::Individual, X::BitMatrix) = evaluate.(Ref(ind), eachcol(X))

"""
    evaluate!(ind::Individual, X::BitMatrix) -> SubArray{Bool}

Evaluate `ind` on `X` and store the result in `ind.ŷ`. Each column of `X` represents a sample.
Meanwhile, the evaluation results are returned as a view of `ind.ŷ`.
"""
function evaluate!(ind::Individual, X::BitMatrix)::SubArray
    num_samples = size(X, 2)
    if length(ind.ŷ) < num_samples
        resize!(ind.ŷ, num_samples)
    end
    ŷ = @view ind.ŷ[1:num_samples]
    # there should be no allocation here
    @allocated ŷ .= evaluate(ind, X)
    return ŷ
end

"""
    count_fitting_error!(ind::Individual, X::BitMatrix, y::BitVector) -> Integer

Count the fitting error of `ind` on `X` and `y`. Each column of `X`` represents a sample.
"""
function count_fitting_error!(ind::Individual, X::BitMatrix, y::BitVector)::Integer
    ŷ = evaluate!(ind, X)
    return count(z -> z[1] != z[2], zip(y, ŷ))
end


"""
    count_complexity(ind::Individual) -> Float64

Count the complexity of `ind`.
"""
function count_complexity(ind::Individual)
    # regulator selection ratio
    selected = mark_selected_regulators!(ind)
    selected_ratio = sum(selected) / get_num_inputs(ind)
    # D-edge activation ratio
    d_ratio = sum(ind.DN) / get_num_conjunctions(ind)
    # C-edge activation ratio
    num_activated_c_edges = 0
    for i in eachindex(ind.DN)
        if ind.DN[i]
            num_activated_c_edges += sum(abs, @view ind.CNs[:, i])
        end
    end
    c_ratio = num_activated_c_edges / length(ind.CNs) 
    return selected_ratio + d_ratio + c_ratio
end


function evaluate_fitness!(ind::Individual, X::BitMatrix, y::BitVector)
    num_errors = count_fitting_error!(ind, X, y)
    ind.fitting_error_rate = num_errors / length(y)
    ind.complexity = count_complexity(ind)
    return nothing
end

"""
    get_fitness(ind::Individual) -> (Float64, Float64)

Get the fitness of `ind` as a tuple of (fitting_error_rate, complexity).
Note that `ind` must be evaluated with `evaluate_fitness!` before calling this function.
"""
function get_fitness(ind::Individual)
    return ind.fitting_error_rate, ind.complexity
end


# function Base.:<(ind1::Individual, ind2::Individual)
#     return (ind1.fitting_error_rate, ind1.complexity) < (ind2.fitting_error_rate, ind2.complexity)
# end

function Base.isless(ind1::Individual, ind2::Individual)
    return (ind1.fitting_error_rate, ind1.complexity) < (ind2.fitting_error_rate, ind2.complexity)
end

