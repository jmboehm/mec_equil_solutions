# main.jl

# (c) 2020 Johannes Boehm <johannes.boehm@sciencespo.fr>

using Revise
using DataFrames

using Random
using Distances

using Test

include("src/JacobiAlgorithm.jl")

# Locations
rng = MersenneTwister(777)
nbz = 3
z_df = DataFrame(h = rand(rng, nbz), v = rand(rng, nbz))

# Supply
nbi = 200
rng = MersenneTwister(778)
i_df = DataFrame(h = rand(rng, nbi), 
    v = rand(rng, nbi),
    τ = rand(rng, nbi),
    λ = 10.0 * rand(rng, nbi)
    )

# Demand
nbj = 500
rng = MersenneTwister(779)
j_df = DataFrame(h = rand(rng, nbj), 
    v = rand(rng, nbj),
    σ = 1.0 ./ rand(rng, nbj),
    ϵ = 20.0 .* rand(rng, nbj),
    η = rand(rng, nbj)
    )

avg_speed_drive = 25.0
avg_speed_walk = 4.0

T_iz = pairwise(Euclidean(), Matrix(i_df[:,[:h,:v]])', Matrix(z_df[:,[:h,:v]])' ) ./ avg_speed_drive
T_jz = pairwise(Euclidean(), Matrix(j_df[:,[:h,:v]])', Matrix(z_df[:,[:h,:v]])' ) ./ avg_speed_walk

function utility_firms(p::R, τ::R, λ::R, T::R) where R<:Real
    return p^(one(R) - τ) - λ * T
end
function utility_consumers(p::R, σ::R, ϵ::R, T::R) where R<:Real
    expo = one(R) - one(R) / σ 
    return ( p^expo + (ϵ*T)^expo )^(one(R)/expo)
end


# it would be more "Julian" to have a separate type for firms

function demand_z(prices_z::Vector{R}, consumers_j::DataFrame, T::Matrix{R}) where R<:Real
    number_of_markets = size(prices_z, 1)
    number_of_consumers = size(consumers_j, 1)
    d = zeros(R, number_of_markets)
    # construct matrix of utilities
    u_jz = zeros(R, number_of_consumers, number_of_markets)
    for z = 1:number_of_markets
        for j = 1:number_of_consumers
            u_jz[j,z] = utility_consumers(prices_z[z],consumers_j[j,:σ],consumers_j[j,:ϵ], T[j,z])
        end
    end
    # for each firm i, increment the supply vector at the position where it has the highest value
    for j=1:number_of_consumers
        min_index = -1
        min = consumers_j[j,:η]
        for z = 1:number_of_markets
            if u_jz[j,z] < min
                min = u_jz[j,z]
                min_index = z 
            end
        end
        if min_index != -1
            d[min_index] += 1
        else 
            # outside option is better
        end
    end
    return d
end
function supply_z(prices_z::Vector{R}, firms_i::DataFrame, T::Matrix{R}) where R<:Real
    number_of_firms = size(firms_i, 1)
    number_of_markets = size(prices_z, 1)
    s = zeros(R, number_of_markets)
    # construct matrix of utilities
    u_iz = zeros(R, number_of_firms, number_of_markets)
    for z = 1:number_of_markets
        for i = 1:number_of_firms
            u_iz[i,z] = utility_firms(prices_z[z],firms_i[i,:τ],firms_i[i,:λ],T[i,z])
        end
    end
    # for each firm i, increment the supply vector at the position where it has the highest value
    for i=1:number_of_firms
        max_index = -1
        max = zero(R)
        for z = 1:number_of_markets
            if u_iz[i,z] > max
                max = u_iz[i,z]
                max_index = z 
            end
        end
        s[max_index] += 1
    end
    return s
end

function ssmooth_z(prices_z::Vector{R}, firms_i::DataFrame, T::Matrix{R}, smoothpar::R) where R<:Real
    number_of_firms = size(firms_i, 1)
    number_of_markets = size(prices_z, 1)
    s = zeros(R, number_of_markets)
    # construct matrix of utilities
    u_iz = zeros(R, number_of_firms, number_of_markets)
    for z = 1:number_of_markets
        for i = 1:number_of_firms
            u_iz[i,z] = utility_firms(prices_z[z],firms_i[i,:τ],firms_i[i,:λ],T[i,z])
        end
    end
    # calculate smooth max
    u_i0 = zeros(R, number_of_firms)
    max_i = [max(zero(R), maximum([u_iz[i,z] for z=1:number_of_markets])) for i=1:number_of_firms]
    utilde_i = max_i .- smoothpar.* log.( exp.( (u_i0 .- max_i)./smoothpar ) .+ sum( exp.( (u_iz .- repeat(max_i,1,number_of_markets))./smoothpar ), dims=2 ) )
    return sum( exp.( ( u_iz .- repeat(utilde_i,1,number_of_markets)) ./ smoothpar) , dims=1)
end
function dsmooth_z(prices_z::Vector{R}, consumers_j::DataFrame, T::Matrix{R}, smoothpar::R) where R<:Real
    # here I denote the cost by u_jz
    number_of_markets = size(prices_z, 1)
    number_of_consumers = size(consumers_j, 1)
    d = zeros(R, number_of_markets)
    # construct matrix of utilities
    u_jz = zeros(R, number_of_consumers, number_of_markets)
    for z = 1:number_of_markets
        for j = 1:number_of_consumers
            u_jz[j,z] = utility_consumers(prices_z[z],consumers_j[j,:σ],consumers_j[j,:ϵ], T[j,z])
        end
    end
    # calculate smooth max
    u_j0 = [consumers_j[j,:η] for j=1:number_of_consumers]
    min_j = [min(u_j0[j], minimum([u_jz[j,z] for z=1:number_of_markets])) for j=1:number_of_consumers]
    utilde_j = min_j .- smoothpar.* log.( exp.( (-u_j0 .+ min_j)./smoothpar ) .+ sum( exp.( (-u_jz .+ repeat(min_j,1,number_of_markets))./smoothpar ), dims=2 ) )
    return sum( exp.( ( -u_jz .+ repeat(utilde_j,1,number_of_markets)) ./ smoothpar) , dims=1)
end
# excess supply function, smoothed
function esmooth_z(prices_z::Vector{R}, firms_i::DataFrame, T_firms::Matrix{R}, consumers_j::DataFrame, T_consumers::Matrix{R}, smoothpar::R) where R<:Real
    return ssmooth_z(prices_z, firms_i, T_firms, smoothpar) .- dsmooth_z(prices_z, consumers_j, T_consumers, smoothpar)
end

# show the validity of the approximations

# exact
supply_z(ones(Float64,3), i_df, T_iz)
# approximation
ssmooth_z(ones(Float64,3), i_df, T_iz, 0.0000001)
# test it
@test maximum(abs.(supply_z(ones(Float64,3), i_df, T_iz) .- ssmooth_z(ones(Float64,3), i_df, T_iz, 0.0000001))) ≈ 0.0 atol= 1e04

# exact
demand_z(ones(Float64,3).*0.1, j_df, T_jz)
# approximation
dsmooth_z(ones(Float64,3).*0.1, j_df, T_jz, 0.0000001)
# test it
@test maximum(abs.(demand_z(ones(Float64,3).*0.1, j_df, T_jz) .- dsmooth_z(ones(Float64,3).*0.1, j_df, T_jz, 0.0000001))) ≈ 0.0 atol= 1e04

include("src/JacobiAlgorithm.jl")

# define the problem
e = JacobiAlgorithm.EquilibriumProblem(nbz, p->esmooth_z(p,i_df, T_iz, j_df, T_jz, 1e-7), 0.0, 1e5, zeros(Float64,nbz), zeros(Float64,nbz) )

# try out excess supply fct
e.e_z(zeros(Float64,3)) .- e.q_z'

# try out coordinate update
p_z = ones(Float64, 3).*0.1
JacobiAlgorithm.coordinate_update(e,1,p_z)
JacobiAlgorithm.coordinate_update(e,2,p_z)
JacobiAlgorithm.coordinate_update(e,3,p_z)

# try out the first iterations of Jacobi algorithm
p_z = ones(Float64, 3).*0.1
for i = 1:5
    global p_z
    p_z , _ = JacobiAlgorithm.fJ_z(e,p_z)
    @show p_z
end

# solve the problem using the jacobi algorithm: about 4.2 seconds
JacobiAlgorithm.solve(e, steptol=1e-12, valtol=1e-5, output=2)

# threaded version: < 2 seconds
JacobiAlgorithm.solve_threaded(e, steptol=1e-12, valtol=1e-5, output=2)