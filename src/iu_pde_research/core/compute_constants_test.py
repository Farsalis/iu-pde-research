"""Python–Julia interoperability test for forcing/constants.

Exercises a Julia script (compute_constants_test.jl) via juliacall to verify
shared forcing or constant definitions. Requires a working Julia environment
and the juliacall package; not part of the standard pytest suite.
"""

from juliacall import Main as jl

# jl is relative to the juliapkg.json file
jl.include("./src/iu_pde_research/core/compute_constants_test.jl")

# Explicitly convert Python floats to Julia Float64s
x_jl = jl.convert(jl.Float64, 2.0)
t_jl = jl.convert(jl.Float64, 1.0)

# Julia now recognizes types
julia_returned_result = jl._default_julia_F(x_jl, t_jl)

print(julia_returned_result)
