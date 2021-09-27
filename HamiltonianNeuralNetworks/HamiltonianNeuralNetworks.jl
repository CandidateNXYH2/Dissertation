module HamiltonianNeuralNetworks

using AutoGrad
using FLoops
using LinearAlgebra

include("NetworkArchitecture.jl")
include("RayTracing.jl")
include("Optim.jl")

export HamiltonianNN, 
	   create,
	   trace_step,
       get_loss,
       get_gradient,
       gradient_step
end