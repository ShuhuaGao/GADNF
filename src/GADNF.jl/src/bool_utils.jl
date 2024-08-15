# symbolic manipulation of Boolean functions using `SymbolicUtils.jl`

using SymbolicUtils, SymbolicUtils.Rewriters

# https://github.com/JuliaSymbolics/SymbolicUtils.jl/blob/master/src/simplify_rules.jl
BOOLEAN_RULES = [
        @rule((true | (~x)) => true)
        @rule(((~x) | true) => true)
        @rule((false | (~x)) => ~x)
        @rule(((~x) | false) => ~x)
        @rule((true & (~x)) => ~x)
        @rule(((~x) & true) => ~x)
        @rule((false & (~x)) => false)
        @rule(((~x) & false) => false)

        @rule(!(~x) & ~x => false)
        @rule(~x & !(~x) => false)
        @rule(!(~x) | ~x => true)
        @rule(~x | !(~x) => true)
        @rule(xor(~x, !(~x)) => true)
        @rule(xor(~x, ~x) => false)

        @rule(~x == ~x => true)
        @rule(~x != ~x => false)
        @rule(~x < ~x => false)
        @rule(~x > ~x => false)

        @rule(~x & ~x => ~x)
        @rule(~x | ~x => ~x)
        @rule((~x2 & ~x3) | (~x3 & ~x4) => ~x3 & (~x2 | ~x4))
]

"""
    simplify(ep::SymbolicUtils.BasicSymbolic{Bool}) -> SymbolicUtils.BasicSymbolic{Bool}

Simplify a Boolean expression `ep`.
"""
function simplify_bool(ep::SymbolicUtils.BasicSymbolic{Bool})
    r = Postwalk(Chain(BOOLEAN_RULES))
    # return r(ep)
    return simplify(ep; expand=true, rewriter=r)
end

"""
    generate_data(f::Function, num_samples::Int) -> (BitMatrix, BitVector)

Given a Boolean function that accepts a vector of length `num_inputs`, generate `num_samples` data points.
The results are returned as a bool matrix and a bool vector.
"""
function generate_bool_data(f::Function, num_inputs::Integer, num_samples::Integer)
    @assert num_samples <= 2^num_inputs
    X = BitMatrix(undef, num_inputs, 2^num_inputs)
    y = BitVector(undef, 2^num_inputs)
    for (i, x) in enumerate(Iterators.product(Iterators.repeated([true, false], num_inputs)...))
        X[:, i] .= x
        y[i] = f(@view X[:, i])
    end
    # choose `num_samples` columns randomly
    indices = shuffle(1:size(X, 2))
    selector = @view indices[1:num_samples]
    return X[:, selector], y[selector]
end
