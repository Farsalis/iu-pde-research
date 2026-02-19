"""
Julia -> Python interoperability test file
"""

ENV["JULIA_PYTHONCALL_EXE"] = "C://Users//scfre//CursorProjects//iu-pde-research//venv//Scripts//python.exe"  # Remove before commit to dev

using PythonCall

export _default_julia_F

np = pyimport("numpy")

function _default_julia_F(x::AbstractFloat, t::AbstractFloat)::AbstractFloat
    return sin(x) * exp(-t)
end

py_array = np.array([1.0, 2.0, 3.0])
println("Numpy array from Julia: ", py_array)