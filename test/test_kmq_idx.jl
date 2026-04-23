using Test

function wrapped_kminusq_index(L)
    [mod1(k - q + L ÷ 2 + 1, L) for k in 1:L, q in 1:L]
end

function wrapped_target_index(ks, k, q)
    Δk = ks[2] - ks[1]
    target = mod(ks[k] - ks[q] + π, 2π) - π
    diffs = abs.(ks .- target)
    idx = argmin(diffs)

    # Resolve the π ≡ -π ambiguity in favour of the stored -π point.
    if isapprox(abs(target), π; atol=Δk / 4)
        idx = 1
    end
    return idx
end

@testset "kmq_idx matches [-π, π) grid subtraction" begin
    for L in (10, 20, 100)
        @test iseven(L)
        Δk = 2π / L
        ks = collect(range(-π, stop=π - Δk, length=L))
        kmq_idx = wrapped_kminusq_index(L)

        # k - k = 0 must map to the zero-momentum grid point.
        zero_idx = L ÷ 2 + 1
        for k in 1:L
            @test kmq_idx[k, k] == zero_idx
        end

        # Exhaustively verify agreement with wrapped subtraction on the grid.
        for k in 1:L, q in 1:L
            @test kmq_idx[k, q] == wrapped_target_index(ks, k, q)
        end
    end
end
