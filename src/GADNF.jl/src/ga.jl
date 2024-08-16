# main loop 

Base.@kwdef struct GAConfig
    num_generations::Int = 100
    population_size::Int = 100
    num_elites::Int = 1
    num_conjunctions::Int = 3

    tournament_size::Int = 2

    CN_subtree_cx_rate::Float64 = 0.05
    edge_cx_rate::Float64 = 0.05
    mut_reduce_rate::Float64 = 0.1
    mut_add_rate::Float64 = 0.1
    mut_rate_min::Float64 = 0.3
    mut_rate_max::Float64 = 0.7
    mut_rate::Float64 = 0.2

    allowed_stagnation_generations::Int = 10
    probabilistic_replacement::Bool=true
end

"""
    adjust_mut_reduce_rate(cfg::GAConfig, g::Integer) -> Float64

Adjust the mutation reduction rate as a linear function of the generation.
As `g` increases, the reduction mutation rate increases.
"""
function adjust_mut_reduce_rate(cfg::GAConfig, g::Integer)
    k = (cfg.mut_rate_max - cfg.mut_rate_min) / cfg.num_generations
    b = cfg.mut_rate_min
    mut_reduce_rate = k * g + b 
    return mut_reduce_rate
end

function report(population, g, progress_report)
    if progress_report
        best = minimum(population)
        @printf("%-5d (%-6.4f, %-6.4f)\n", g, best.fitting_error_rate, best.complexity)
    end
end


function run_GA(X::BitMatrix, y::BitVector; cfg::GAConfig, target::String="y", 
        features::AbstractVector{String}=String[],
        progress_report::Bool = true, history_jld2::String="")
    num_inputs, num_samples = size(X)
    @assert num_samples == length(y)
    @assert iseven(cfg.population_size)
    if isempty(features)
        features = ["x$i" for i in 1:num_inputs]
    end
    @assert num_inputs == length(features)
    println("Running GA for $target ...")

    population = [Individual(num_inputs, cfg.num_conjunctions) for _ in 1:cfg.population_size]
    # create the new population in advance to reduce memory allocations
    new_population = [Individual(num_inputs, cfg.num_conjunctions) for _ in 1:cfg.population_size]
    # evaluate initial population
    evaluate_fitness!(population, X, y)

    progress_report && @printf("%-5s %-8s %-8s\n", "Gen", "Error", "Complexity")
    history = isempty(history_jld2) ? nothing : [deepcopy(population)]
    store(population, history) = !isnothing(history) && push!(history, deepcopy(population))

    current_best = minimum(population)
    num_stagnations = 0

    for g in 1:cfg.num_generations
        # selection
        select_tournament!(new_population, population; tournament_size=cfg.tournament_size)
        # update rate parameters
        mut_reduce_rate = adjust_mut_reduce_rate(cfg, g)
        # crossover
        for i in 1:2:cfg.population_size
            crossover!(new_population[i], new_population[i+1]; 
                CN_subtree_cx_rate=cfg.CN_subtree_cx_rate, edge_cx_rate=cfg.edge_cx_rate)
        end
        # mutation
        # mutate!.(new_population; mut_reduce_rate, mut_add_rate)
        mutate2!.(new_population; mut_reduce_rate, mut_rate=cfg.mut_rate)
        # copy elites
        if cfg.num_elites > 0
            select_elites!(@view(new_population[1:cfg.num_elites]), population)
        end
        # evaluate, but no need to evaluate the elites 
        evaluate_fitness!(new_population, X, y)
        new_population, population = population, new_population
        # report
        report(population, g, progress_report)
        # store
        store(population, history)
        # early termination
        new_best = minimum(population)
        if new_best < current_best
            current_best = new_best
            num_stagnations = 0
        else
            num_stagnations += 1
        end
        if num_stagnations >= cfg.allowed_stagnation_generations
            println("Early termination: $num_stagnations generations without improvement.")
            break
        end
    end

    if !isempty(history_jld2)
        JLD2.jldsave(history_jld2; history, cfg, target, features)
    end

    return population
end


"""
Genotype distance between two individuals calculated as the Hamming distance.
"""
function compute_distance(ind1::Individual, ind2::Individual)::Int
    d = 0
    for i in 1:get_num_conjunctions(ind1)
        for j in 1:get_num_inputs(ind1)
            v1 = ind1.DN[i] ? ind1.CNs[j, i] : 0
            v2 = ind2.DN[i] ? ind2.CNs[j, i] : 0
            d += v1 != v2
        end
    end
    return d 
end


function pick_winner_deterministic(tournament)::Individual
    if tournament[1] < tournament[2]
        return tournament[1]
    elseif tournament[2] < tournament[1]
        return tournament[2]
    else
        return rand(tournament)
    end
end

function pick_winner_probabilistic(tournament)::Individual
    pick(f1, f2) = rand() > (f1 / (f1 + f2 + 1e-5)) ? 1 : 2      # smaller the better
    if rand() < 1   # which criterion for comparison?
        w = pick(tournament[1].fitting_error_rate, tournament[2].fitting_error_rate)
    else
        w = pick(tournament[1].complexity, tournament[2].complexity)
    end
    return tournament[w]
end

"""
    run_CGA

Crowding GA. 
TODO: not changed for combined objective yet.
"""
function run_CGA(X::BitMatrix, y::BitVector; cfg::GAConfig, target::String="y", 
    features::AbstractVector{String}=String[], 
    progress_report::Bool = true, history_jld2::String="")

    num_inputs, num_samples = size(X)
    @assert num_samples == length(y)
    @assert iseven(cfg.population_size)
    if isempty(features)
        features = ["x$i" for i in 1:num_inputs]
    end
    @assert num_inputs == length(features)
    println("Running GA for $target ...")

    old_population = [Individual(num_inputs, cfg.num_conjunctions) for _ in 1:cfg.population_size]
    # evaluate initial population
    evaluate_fitness!.(old_population, Ref(X), Ref(y))
    new_population = [Individual(num_inputs, cfg.num_conjunctions) for _ in 1:cfg.population_size] # placeholder
    children = [Individual(num_inputs, cfg.num_conjunctions) for _ in 1:2] # placeholder
    parents = deepcopy(children)
    distances = zeros(Int, 2, 2)
    index_pool = shuffle(1:cfg.population_size)

    current_best = minimum(old_population)
    num_stagnations = 0

    for g in 1:cfg.num_generations
        shuffle!(index_pool)
        # update rate parameters
        mut_reduce_rate = adjust_mut_reduce_rate(cfg, g)
        for cursor in 1:2:cfg.population_size
            # Phase 1: choose two parents randomly for 2-tournament
            parents .= (old_population[index_pool[cursor]], old_population[index_pool[cursor+1]])
            # Phase 2: cx and mutate
            copy!.(children, parents)    # keep parents unmodified
            crossover!(children[1], children[2]; 
                    CN_subtree_cx_rate=cfg.CN_subtree_cx_rate, edge_cx_rate=cfg.edge_cx_rate)
            mutate2!.(children; mut_reduce_rate, mut_rate=cfg.mut_rate)
            evaluate_fitness!.(children, Ref(X), Ref(y))
            # Phase 3: compute distances between parents and children
            for i in 1:2, j in 1:2
                distances[i, j] = compute_distance(parents[i], children[j])
            end
            # Phase 4: match
            d1 = distances[1, 1] + distances[2, 2]
            d2 = distances[1, 2] + distances[2, 1]
            if d1 < d2
                m = ((parents[1], children[1]), (parents[2], children[2]))
            else
                m = ((parents[1], children[2]), (parents[2], children[1]))
            end 
            # Phase 5: replacement
            for (i, tournament) in enumerate(m)
                if cfg.probabilistic_replacement
                    winner = pick_winner_probabilistic(tournament)
                else
                    winner = pick_winner_deterministic(tournament)
                end
                copy!(new_population[cursor + i - 1], winner)
            end
        end
        # elitism
        if cfg.num_elites > 0
            select_elites!(@view(new_population[1:cfg.num_elites]), old_population)
        end
        # exchange for the next generation
        old_population, new_population = new_population, old_population
        if progress_report
            report(old_population, g, progress_report)
        end
        # early termination
        new_best = minimum(old_population)
        if new_best < current_best
            current_best = new_best
            num_stagnations = 0
        else
            num_stagnations += 1
        end
        if num_stagnations >= cfg.allowed_stagnation_generations
            println("Early termination: $num_stagnations generations without improvement.")
            break
        end
    end

    return old_population
end