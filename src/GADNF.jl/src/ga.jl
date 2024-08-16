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
end

"""
    adjust_mut_reduce_rate(cfg::GAConfig, g::Integer) -> Float64

Adjust the mutation reduction rate as a linear function of the generation.
As `g` increases, the reduction mutation rate increases.
"""
function adjust_mut_rate(cfg::GAConfig, g::Integer)
    k = (cfg.mut_rate_max - cfg.mut_rate_min) / cfg.num_generations
    b = cfg.mut_rate_min
    mut_reduce_rate = k * g + b 
    mut_add_rate = cfg.mut_rate_max + cfg.mut_rate_min - mut_reduce_rate
    return mut_reduce_rate, mut_add_rate
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
    evaluate_fitness!.(population, Ref(X), Ref(y))


    @printf("%-5s %-8s %-8s\n", "Gen", "Error", "Complexity")
    function report(population, g)
        if progress_report
            best = minimum(population)
            @printf("%-5d (%-6.4f, %-6.4f)\n", g, best.fitting_error_rate, best.complexity)
        end
    end
    
    history = isempty(history_jld2) ? nothing : [deepcopy(population)]
    store(population, history) = !isnothing(history) && push!(history, deepcopy(population))

    current_best = minimum(population)
    num_stagnations = 0

    for g in 1:cfg.num_generations
        # selection
        select_tournament!(new_population, population; tournament_size=cfg.tournament_size)
        # update rate parameters
        mut_reduce_rate, mut_add_rate = adjust_mut_rate(cfg, g)
        mut_add_rate = 0.1
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
        evaluate_fitness!.(@view(new_population[cfg.num_elites+1:end]), Ref(X), Ref(y))
        new_population, population = population, new_population
        # report
        report(population, g)
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