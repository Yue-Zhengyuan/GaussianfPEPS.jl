using Test
using SafeTestsets

@time @safetestset "Expectation values in paired state" begin
    include("paired_state.jl")
end
@time @safetestset "Correlation matrix" begin
    include("cormat.jl")
end
@time @safetestset "Bogoliubov transformation" begin
    include("bogoliubov.jl")
end
@time @safetestset "BCS mean field Hamiltonian" begin
    include("bcs.jl")
end
@time @safetestset "Translation to fPEPS" begin
    include("translate.jl")
end
