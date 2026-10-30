using LinearAlgebra
using Printf
using Sunny

include(joinpath(@__DIR__, "triangular_ladder.jl"))

# Parameters use [J1, Δ, Jpmpm, Jzpm, J2].
# The saturation-field helper instead expects [J1, Jzz, Jpmpm, Jzpm, J2].
function saturation_parameters(Js)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    return [J1, J1 * Δ, Jpmpm, Jzpm, J2]
end

"""
Create the same g=1, field-along-+x system used in FIM_check.jl.
The spins point along -x.
"""
function make_fim_system(Js, B; dims=(1, 1, 1))
    length(Js) == 5 || throw(ArgumentError("Expected five parameters"))
    all(isfinite, Js) || throw(ArgumentError("Parameters must be finite"))
    isfinite(B) && B > 0 || throw(ArgumentError("B must be finite and positive"))

    #J1, Δ, Jpmpm, Jzpm, J2 = Js
    J1, Jzz, Jpmpm, Jzpm, J2 = Js
    cryst = triangular_crystal()

    sys = System(
        cryst,
        [1 => Moment(s=1/2, g=1)],
        :dipole;
        dims=dims,
        seed=0,
    )

    # Jten = [
    #     J1 + 2Jpmpm  0.0          Jzpm
    #     0.0           J1 - 2Jpmpm  0.0
    #     Jzpm          0.0          J1 * Δ
    # ]
    Jten = [
        J1 + 2Jpmpm  0.0          Jzpm
        0.0           J1 - 2Jpmpm  0.0
        Jzpm          0.0          Jzz
    ]

    set_exchange!(sys, Jten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0.0, 0.0])
    polarize_system!(sys, [-1.0, 0.0, 0.0])

    return sys
end

"""
Forward observable, matching the :perp behavior in FIM_check.jl.

LSWT:   [vec(dispersion); vec(intensity)]
Ladder: vec(energy-resolved perpendicular intensity)
"""
function spectrum_vector(Js, B, method;
    dims=(1, 1, 1),
    Nk=12,
    Qz=1.0,
    eta=0.2,
    energies=collect(0.0:0.01:10.0),
)
    sys = make_fim_system(Js, B; dims=dims)
    path = q_space_path(
        sys.crystal,
        [[0.0, 0.0, Qz], [1.0, 1.0, Qz]],
        Nk,
    )

    values = if method === :lswt
        swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
        res = intensities_bands(swt, path)
        vcat(vec(res.disp), vec(res.data))

    elseif method === :ladder
        swt = SpinWaveTheory(sys; measure=dssf_measure(sys))

        # Parallelism is across sweep jobs; avoid nested ladder threading.
        data, nbreak = ladder_dssf(
            swt,
            path;
            energies=energies,
            η=eta,
            threaded=false,
            verbose=false,
        )

        nbreak == 0 ||
            error("Ladder breakdown at $nbreak momenta: Js=$Js, B=$B")

        vec(data[7, :, :])

    else
        throw(ArgumentError("Unknown method: $method"))
    end

    result = Float64.(values)
    all(isfinite, result) ||
        error("Nonfinite spectrum: method=$method, Js=$Js, B=$B")

    return result
end

"""
Compute F = J'J using central finite differences.

The field, momentum path and energy grid remain fixed for every derivative.
The FIM condition number is (σmax(J) / σmin(J))².

Numerically rank-deficient Jacobians produce condition = Inf.
"""
function fim_diagnostics(Js0, B, method;
    delta=1e-3,
    rank_rtol=nothing,
    kwargs...,
)
    delta > 0 && isfinite(delta) ||
        throw(ArgumentError("delta must be finite and positive"))

    Js = Float64.(Js0)
    d = length(Js)
    steps = delta .* max.(abs.(Js), 1.0)

    # Initialize the Jacobian from the first derivative. A baseline
    # spectrum is unnecessary for central finite differences.
    J = Matrix{Float64}(undef, 0, d)

    for p in 1:d
        plus = copy(Js)
        minus = copy(Js)
        plus[p] += steps[p]
        minus[p] -= steps[p]

        Sp = spectrum_vector(plus, B, method; kwargs...)
        Sm = spectrum_vector(minus, B, method; kwargs...)

        length(Sp) == length(Sm) ||
            error("Observable size changed between perturbations")

        if p == 1
            J = Matrix{Float64}(undef, length(Sp), d)
        end

        length(Sp) == size(J, 1) ||
            error("Observable size changed between parameters")

        @views J[:, p] .= (Sp .- Sm) ./ (2 * steps[p])
    end

    all(isfinite, J) || error("Nonfinite Jacobian")

    # SVD avoids extracting a tiny eigenvalue from the explicitly
    # formed Gram matrix, which can lose accuracy near singularity.
    singular_values = svdvals(J)  # Descending order.
    length(singular_values) == d ||
        error("Too few observables for a $d-parameter FIM")

    rtol = isnothing(rank_rtol) ?
        max(size(J)...) * eps(Float64) : Float64(rank_rtol)

    isfinite(rtol) && 0 <= rtol < 1 ||
        throw(ArgumentError("rank_rtol must lie in [0, 1)"))

    smax = first(singular_values)
    smin = last(singular_values)
    tolerance = rtol * smax
    numerical_rank = count(s -> s > tolerance, singular_values)

    raw_condition = smin > 0 ? (smax / smin)^2 : Inf
    condition = numerical_rank == d ? raw_condition : Inf

    return (
        condition=condition,
        raw_condition=raw_condition,
        numerical_rank=numerical_rank,
        singular_values=singular_values,
        eigenvalues=singular_values .^ 2,
        F=Symmetric(J' * J),
        steps=steps,
        n_observables=size(J, 1),
    )
end

function print_condition_table(io, conditions, methods, jzpm_values)
    @printf(io, "%-10s", "method")
    for value in jzpm_values
        @printf(io, " %13.6g", value)
    end
    println(io)

    for (row, method) in enumerate(methods)
        @printf(io, "%-10s", string(method))
        for col in eachindex(jzpm_values)
            @printf(io, " %13.6e", conditions[row, col])
        end
        println(io)
    end
end

# Escape diagnostic messages for CSV, including multiline exceptions.
csv_quote(value) = "\"" * replace(string(value), "\"" => "\"\"") * "\""

function save_results(outdir, conditions, methods, jzpm_values, records)
    mkpath(outdir)

    table_path = joinpath(outdir, "condition_numbers.csv")
    open(table_path, "w") do io
        println(io, "method,", join(string.(jzpm_values), ","))
        for (row, method) in enumerate(methods)
            println(
                io,
                string(method), ",",
                join(string.(conditions[row, :]), ","),
            )
        end
    end

    diagnostics_path = joinpath(outdir, "diagnostics.csv")
    open(diagnostics_path, "w") do io
        println(
            io,
            "method,J1,Delta,Jpmpm,Jzpm,J2,B,delta,Nk,Qz,eta," *
            "energy_min,energy_step,energy_max,condition,raw_condition," *
            "rank,n_observables,sigma_max,sigma_min,seconds,status,error",
        )

        for r in records
            diag = r.diagnostics

            fields = [
                string(r.method),
                string.(r.Js)...,
                string(r.B),
                string(r.delta),
                string(r.Nk),
                string(r.Qz),
                string(r.eta),
                string(r.energy_min),
                string(r.energy_step),
                string(r.energy_max),
                string(isnothing(diag) ? NaN : diag.condition),
                string(isnothing(diag) ? NaN : diag.raw_condition),
                string(isnothing(diag) ? 0 : diag.numerical_rank),
                string(isnothing(diag) ? 0 : diag.n_observables),
                string(isnothing(diag) ? NaN : first(diag.singular_values)),
                string(isnothing(diag) ? NaN : last(diag.singular_values)),
                string(r.seconds),
                isnothing(diag) ? "failed" : "ok",
                csv_quote(r.error),
            ]

            println(io, join(fields, ","))
        end
    end

    return table_path, diagnostics_path
end

"""
Run the 2×7 sweep.

Rows:    ladder, lswt
Columns: Jzpm_values, in the supplied order

fixed_B=nothing:
    Choose B = B_sat + abs(J1) separately at each column.
fixed_B=<number>:
    Use that field for every column and both methods.

Failed calculations are recorded as NaN, with details in diagnostics.csv.
Rank-deficient calculations are recorded as Inf.
"""
function run_condition_sweep(;
    base_Js=[1.0, 0.7, 0.2, 0.2, 0.051],
    jzpm_values=[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
    fixed_B=nothing,
    delta=1e-3,
    dims=(1, 1, 1),
    Nk=60,
    Qz=0.0,
    eta=0.2,
    energy_step=0.01,
    energy_maximum=20.0,
    rank_rtol=nothing,
    threaded=true,
    outdir=joinpath(@__DIR__, "..", "data"),
)
    length(base_Js) == 5 ||
        throw(ArgumentError("base_Js must contain five parameters"))
    all(isfinite, base_Js) ||
        throw(ArgumentError("base_Js must be finite"))

    jzpm_values = Float64.(collect(jzpm_values))
    !isempty(jzpm_values) && all(isfinite, jzpm_values) ||
        throw(ArgumentError("jzpm_values must be nonempty and finite"))

    Nk >= 2 || throw(ArgumentError("Nk must be at least 2"))
    isfinite(eta) && eta > 0 ||
        throw(ArgumentError("eta must be finite and positive"))
    isfinite(energy_step) && 0 < energy_step <= eta / 2 ||
        throw(ArgumentError("Require 0 < energy_step <= eta/2"))
    isfinite(energy_maximum) && energy_maximum >= energy_step ||
        throw(ArgumentError("Invalid energy_maximum"))

    energies = collect(0.0:energy_step:energy_maximum)
    methods = (:ladder, :lswt)

    parameter_sets = [
        [base_Js[1], base_Js[2], base_Js[3], value, base_Js[5]]
        for value in jzpm_values
    ]

    # Select fields before the threaded numerical work.
    # Conversion Δ -> Jzz is essential for the helper.
    fields = Vector{Float64}(undef, length(jzpm_values))
    for col in eachindex(jzpm_values)
        if isnothing(fixed_B)
            helper_Js = parameter_sets[col] #saturation_parameters(parameter_sets[col])
            B, _, _ = polarized_system(helper_Js, [1.0, 0.0, 0.0])
            fields[col] = B
        else
            fields[col] = Float64(fixed_B)
        end

        isfinite(fields[col]) && fields[col] > 0 ||
            throw(ArgumentError("Invalid measurement field"))

        @printf(
            "Jzpm=% .3f: B=%.8g\n",
            jzpm_values[col], fields[col],
        )
    end

    conditions = fill(NaN, length(methods), length(jzpm_values))
    jobs = collect(CartesianIndices(conditions))
    records = Vector{Any}(undef, length(jobs))

    function run_job(k)
        row, col = Tuple(jobs[k])
        method = methods[row]
        Js = parameter_sets[col]
        B = fields[col]

        started = time_ns()
        diagnostics = nothing
        error_message = ""

        try
            diagnostics = fim_diagnostics(
                Js, B, method;
                delta=delta,
                rank_rtol=rank_rtol,
                dims=dims,
                Nk=Nk,
                Qz=Qz,
                eta=eta,
                energies=energies,
            )
            conditions[row, col] = diagnostics.condition
        catch err
            err isa InterruptException && rethrow()
            error_message = sprint(showerror, err, catch_backtrace())
        end

        records[k] = (
            method=method,
            Js=copy(Js),
            B=B,
            delta=delta,
            Nk=Nk,
            Qz=Qz,
            eta=eta,
            energy_min=first(energies),
            energy_step=energy_step,
            energy_max=last(energies),
            diagnostics=diagnostics,
            seconds=(time_ns() - started) / 1e9,
            error=error_message,
        )

        return nothing
    end

    # One BLAS thread per job avoids oversubscribing CPU cores.
    previous_blas_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)

        if threaded && Threads.nthreads() > 1
            Threads.@threads :dynamic for k in eachindex(jobs)
                run_job(k)
            end
        else
            for k in eachindex(jobs)
                run_job(k)
            end
        end
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end

    table_path, diagnostics_path = save_results(
        outdir, conditions, methods, jzpm_values, records,
    )

    println("\nFIM condition numbers:")
    print_condition_table(stdout, conditions, methods, jzpm_values)

    println("\nTable:       ", table_path)
    println("Diagnostics: ", diagnostics_path)

    failures = count(r -> isnothing(r.diagnostics), records)
    if failures > 0
        @warn "$failures calculations failed; see diagnostics.csv. Failed cells contain NaN."
    end

    return (
        conditions=conditions,
        methods=methods,
        jzpm_values=jzpm_values,
        fields=fields,
        records=records,
        table_path=table_path,
        diagnostics_path=diagnostics_path,
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_condition_sweep()
end
