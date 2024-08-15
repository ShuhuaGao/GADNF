# selection during evolution

"""
    Base.copy!(ind1::Individual, ind2::Individual)

Deep copy `ind2` into `ind1`.
"""
function Base.copy!(ind1::Individual, ind2::Individual)
    copy!(ind1.DN, ind2.DN)
    copy!(ind1.CNs, ind2.CNs)
    ind1.fitting_error_rate = ind2.fitting_error_rate
    ind1.complexity = ind2.complexity
    copy!(ind1.ŷ, ind2.ŷ)
    copy!(ind1.regulator_selected, ind2.regulator_selected)
    return ind1
end


function select_tournament(ind1::Individual, ind2::Individual)::Individual
    return ind1 < ind2 ? ind1 : ind2 
end

"""
    select_tournament!(new_population::AbstractVector{Individual}, old_population::AbstractVector{Individual}; 
        tournament_size=2)

Form a new population in place by selecting individuals from the old population using tournament selection.
Note that the size of `new_population` and `old_population` may differ.
The individual in `new_population` is deeply copied and independent from that in `old_population`.
"""
function select_tournament!(new_population::AbstractVector{Individual}, old_population::AbstractVector{Individual}; 
        tournament_size=2)
    @assert tournament_size >= 2
    if tournament_size== 2
        for i in eachindex(new_population)
            winner = select_tournament(old_population[rand(1:length(old_population))], old_population[rand(1:length(old_population))])
            copy!(new_population[i], winner)   # we need a deep copy
        end
    else
        error("Not implemented yet")
    end
    return nothing
end

"""
    select_elites!(elites::AbstractVector{Individual}, old_population::AbstractVector{Individual})

Select elites from `old_population` and store them in `elites`. Deep copy is made.
Note that, if the length of `elite` ≥ 2, then `old_population` will be sorted in place.
"""
function select_elites!(elites::AbstractVector{Individual}, old_population::AbstractVector{Individual})
    @assert length(elites) >= 1
    if length(elites) == 1
        copy!(elites[1], minimum(old_population))
    else
        sort!(old_population)   # ascending order
        copy!.(elites, @view old_population[1:length(elites)])
    end
    return nothing
end