module JacobiAlgorithm

    struct EquilibriumProblem{T<:Real}
        nbz::Int64 
        e_z::Function 
        pmin::T 
        pmax::T
        q_z::Vector{T}
        p0_z::Vector{T}
        function EquilibriumProblem(nbz::Int64,e_z::Function,pmin::T,pmax::T,q_z::Vector{T},p0_z::Vector{T})  where {T<:Real}
            this = new{T}(nbz,e_z,pmin,pmax,q_z,p0_z) #,NaN,[],[],[],NaN)
        end
    end

    function bisection(f::Function, a::T, b::T;
        tol::T=1e-5, maxiter::Int64=100) where T<:Real
        fa = f(a)
        fa*f(b) <= 0 || error("No real root in [a,b]")
        i = 0
        local c
        while b-a > tol
                i += 1
                i != maxiter || error("Max iteration exceeded")
                c = (a+b)/2
                fc = f(c)
            if fc == 0
                break
            elseif fa*fc > 0
                a = c  # Root is in the right half of [a,b].
                fa = fc
            else
                b = c  # Root is in the left half of [a,b].
            end
        end
        return c
    end

    function coordinate_update(problem::EquilibriumProblem{T}, z::Int64, price_z::Vector{T}) where T<:Real
        old_price = price_z[z]
        f(p) = problem.e_z(setindex!(price_z,p,z))[z] - problem.q_z[z]
        new_price = bisection(f, problem.pmin, problem.pmax,tol=1e-5,maxiter=1000)
        price_z[z] = old_price
        return new_price, new_price - old_price
    end

    function fJ_z(problem::EquilibriumProblem{T}, p_z::Vector{T}) where T<:Real
        nbz = size(p_z,1)
        pp_z = zeros(T, nbz)
        diffp_z = zeros(T, nbz)
        for z = 1:nbz
            pp_z[z], diffp_z[z] = coordinate_update(problem,z,p_z)
        end
        return pp_z, diffp_z
    end

    function fJ_threaded_z(problem::EquilibriumProblem{T}, p_z::Vector{T}) where T<:Real
        nbz = size(p_z,1)
        pp_z = zeros(T, nbz)
        diffp_z = zeros(T, nbz)
        Threads.@threads for z = 1:nbz
            p_z_safe = deepcopy(p_z)
            pp_z[z], diffp_z[z] = coordinate_update(problem,z,p_z_safe)
        end
        return pp_z, diffp_z
    end

    function solve(problem::EquilibriumProblem{T}; 
        maxit::Int64 = 10_000,
        method::Symbol = :jacobian,
        valtol::T = 1e-5,
        steptol::T = 1e-9,
        output::Int64 = 0) where T<:Real

        code = 0
        start_time = time()
        delta_z = zeros(T,problem.nbz)
        diffp_z = zeros(T,problem.nbz)
        iterations = 0
        p_z = deepcopy(problem.p0_z)
        if method != :jacobian
            error("Method not implemented")
        end
        for i = 1:maxit
            p_z, diffp_z = fJ_z(problem, p_z)
            delta_z = problem.e_z(p_z) .- problem.q_z'
            if output>1
                println("p = $p_z")
            end
            if maximum(abs.(delta_z)) < valtol
                code = 0
                iterations = i
                break
            end
            if maximum(abs.(diffp_z)) < steptol
                code = 1
                iterations = i
                break
            end
            code = 2
        end
        comp_time = time() - start_time
        if output > 0
            println("$(String(method)) method converged in $iterations iterations.")
            println("Value of p = $(p_z)")
            println("Value of p' - p = $(diffp_z)")
            println("Value of e(p)-q = $(delta_z)")
            println("Time elapsed: $comp_time")
            println("Code: $code")
        end
        return code
    end

    function solve_threaded(problem::EquilibriumProblem{T}; 
        maxit::Int64 = 10_000,
        method::Symbol = :jacobian,
        valtol::T = 1e-5,
        steptol::T = 1e-9,
        output::Int64 = 0) where T<:Real

        code = 0
        start_time = time()
        delta_z = zeros(T,problem.nbz)
        diffp_z = zeros(T,problem.nbz)
        iterations = 0
        p_z = deepcopy(problem.p0_z)
        if method != :jacobian
            error("Method not implemented")
        end
        for i = 1:maxit
            p_z, diffp_z = fJ_threaded_z(problem, p_z)
            delta_z = problem.e_z(p_z) .- problem.q_z'
            if output>1
                println("p = $p_z")
            end
            if maximum(abs.(delta_z)) < valtol
                code = 0
                iterations = i
                break
            end
            if maximum(abs.(diffp_z)) < steptol
                code = 1
                iterations = i
                break
            end
            code = 2
        end
        comp_time = time() - start_time
        if output > 0
            println("$(String(method)) method converged in $iterations iterations.")
            println("Value of p = $(p_z)")
            println("Value of p' - p = $(diffp_z)")
            println("Value of e(p)-q = $(delta_z)")
            println("Time elapsed: $comp_time")
            println("Code: $code")
        end
        return code
    end
end