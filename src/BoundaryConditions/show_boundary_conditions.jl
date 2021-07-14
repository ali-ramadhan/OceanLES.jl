import Base: show
import Oceananigans: short_show

const DFBC = DefaultPrognosticFieldBoundaryCondition

bc_str(::FBC)     = "Flux"
bc_str(::PBC)     = "Periodic"
bc_str(::OBC)     = "Open"
bc_str(::VBC)     = "Value"
bc_str(::GBC)     = "Gradient"
bc_str(::ZFBC)    = "ZeroFlux"
bc_str(::DFBC)    = "Default"
bc_str(::Nothing) = "Nothing"

#####
##### BoundaryCondition
#####

print_condition(n::Union{Nothing, Number}) = "$n"
print_condition(A::AbstractArray) = "$(Base.dims2string(size(A))) $(typeof(A))"
print_condition(bf::Union{DiscreteBoundaryFunction, ContinuousBoundaryFunction}) = print_condition(bf.func)

function print_condition(f::Function)
    ms = methods(f).ms
    length(ms) == 1 && return "$(ms[1])"
    return "$(ms)"
end

show(io::IO, bc::BoundaryCondition) =
    print(io, "BoundaryCondition: classification=$(bc_str(bc)), condition=$(print_condition(bc.condition))")

#####
##### FieldBoundaryConditions
#####

short_show(fbcs::FieldBoundaryConditions) =
    string("west=$(bc_str(fbcs.west)), ",
           "east=$(bc_str(fbcs.east)), ",
           "south=$(bc_str(fbcs.south)), ",
           "north=$(bc_str(fbcs.north)), ",
           "bottom=$(bc_str(fbcs.bottom)), ",
           "top=$(bc_str(fbcs.top)), ",
           "immersed=$(bc_str(fbcs.immersed))")

show_field_boundary_conditions(bcs::FieldBoundaryConditions, padding="") =
    string("Oceananigans.FieldBoundaryConditions, with boundary conditions", '\n',
           padding, "├── west: ",     typeof(bcs.west), '\n',
           padding, "├── east: ",     typeof(bcs.east), '\n',
           padding, "├── south: ",    typeof(bcs.south), '\n',
           padding, "├── north: ",    typeof(bcs.north), '\n',
           padding, "├── bottom: ",   typeof(bcs.bottom), '\n',
           padding, "├── top: ",      typeof(bcs.top), '\n',
           padding, "└── immersed: ", typeof(bcs.immersed))

Base.show(io::IO, fieldbcs::FieldBoundaryConditions) = print(io, show_field_boundary_conditions(fieldbcs))
