using Test
using KadanoffBaym

# Mirror the storage-access pattern used in main.jl.
@inline kbe_storage_tt(gf, t, t′) = @view gf.data[:, t, t′]

@testset "GreenFunction storage persistence" begin
    gf = GreenFunction(zeros(ComplexF64, 4, 1, 1), SkewHermitian)

    # Wrapper-style indexing returns a copy, so mutating it must not persist.
    slice_copy = gf[1, 1]
    slice_copy[1] = 1 + 0im
    @test gf.data[1, 1, 1] == 0 + 0im

    # Direct view into storage must persist.
    storage_view = kbe_storage_tt(gf, 1, 1)
    storage_view[1] = 2 + 0im
    @test gf.data[1, 1, 1] == 2 + 0im

    copyto!(storage_view, ComplexF64[3, 4, 5, 6])
    @test gf.data[:, 1, 1] == ComplexF64[3, 4, 5, 6]
end

@testset "Safe write pattern after resize!" begin
    gf = GreenFunction(zeros(ComplexF64, 3, 1, 1), SkewHermitian)

    # Write before resize through a fresh storage view.
    copyto!(kbe_storage_tt(gf, 1, 1), ComplexF64[1, 2, 3])
    @test gf.data[:, 1, 1] == ComplexF64[1, 2, 3]

    # After resizing, reacquire the view and then write.
    resize!(gf, 2)
    @test size(gf.data) == (3, 2, 2)

    copyto!(kbe_storage_tt(gf, 2, 1), ComplexF64[7, 8, 9])
    @test gf.data[:, 2, 1] == ComplexF64[7, 8, 9]

    # Wrapper indexing remains non-persistent even after resize.
    resized_slice_copy = gf[2, 1]
    resized_slice_copy[1] = 99 + 0im
    @test gf.data[1, 2, 1] == 7 + 0im
end
