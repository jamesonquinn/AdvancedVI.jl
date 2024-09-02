
module AdvancedVIEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme
    using AdvancedVI
    using AdvancedVI: ADTypes, DiffResults
else
    using ..Enzyme
    using ..AdvancedVI
    using ..AdvancedVI: ADTypes, DiffResults
end

function AdvancedVI.stop_gradient(::ADTypes.AutoEnzyme, x)
    throw("Score function estimator with Enzyme is not supported yet.")
end

function AdvancedVI.restructure_ad_forward(::ADTypes.AutoEnzyme, restructure, params)
    return restructure(params)::typeof(restructure.model)
end

# Enzyme doesn't support f::Bijectors (see https://github.com/EnzymeAD/Enzyme.jl/issues/916)
function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoEnzyme, f, x::AbstractVector{<:Real}, out::DiffResults.MutableDiffResult
)
    Enzyme.API.runtimeActivity!(true)
    ∇x = DiffResults.gradient(out)
    fill!(∇x, zero(eltype(∇x)))
    _, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, ∇x)
    )
    DiffResults.value!(out, y)
    return out
end

function AdvancedVI.value_and_gradient!(
    ::ADTypes.AutoEnzyme,
    f,
    x::AbstractVector{<:Real},
    aux,
    out::DiffResults.MutableDiffResult,
)
    Enzyme.API.runtimeActivity!(true)
    ∇x = DiffResults.gradient(out)
    fill!(∇x, zero(eltype(∇x)))
    _, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Enzyme.Const(f),
        Enzyme.Active,
        Enzyme.Duplicated(x, ∇x),
        Enzyme.Const(aux),
    )
    DiffResults.value!(out, y)
    return out
end

end
