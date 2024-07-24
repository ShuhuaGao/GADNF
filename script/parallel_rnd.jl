using Random 


function xorshiro_par(seed=0)
    X = zeros(Int16, 4, 5)
    Threads.@threads for i = 1:5
        Random.seed!(seed + i)
        X[:, i] .= rand(Int16, 4)
    end
    return X
end


function xorshiro_par2(seed=0)
    X = zeros(Int16, 4, 5)
    Random.seed!(seed)
    Threads.@threads for i = 1:5
        X[:, i] .= rand(Int16, 4)
    end
    return X
end


X = xorshiro_par(0)
display(X)

X2 = xorshiro_par2(9)
display(X2)