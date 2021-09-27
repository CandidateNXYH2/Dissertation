using Distributions: Uniform, Normal
                                    """ Network activation functions """ 
function logistic(x::Float32)
    return 1 / (1 + exp(-x))
end

function logistic(x::AutoGrad.Bcasted{AutoGrad.Result{Vector{Float32}}})
    return 1 / (1 + exp(-x))
end


function leaky_softplus(x::Float32, β::Float32)
    return β*x + (1-β)*x*logistic(x)
end

function leaky_softplus(x::AutoGrad.Bcasted{AutoGrad.Result{Vector{Float32}}}, β::Float32)
    return β*x + (1-β)*x*logistic(x)
end

function α(x::Float32)
    return leaky_softplus(x, 0.1f0)
end

function α(x::AutoGrad.Bcasted{AutoGrad.Result{Vector{Float32}}})
    return leaky_softplus(x, 0.1f0)
end


                                  """ Defining the NN Architecture and Functions """ 

# Struct definition
mutable struct HamiltonianNN
    num_maps::Int64
    
    weights::Vector{Param{Matrix{Float32}}}
     biases::Vector{Param{Vector{Float32}}}
    α::Function

    normalise_inputs::Bool
         use_physics::Bool

    training_min::Vector{Float32}
    training_max::Vector{Float32}
end

# Constructors and initialisors
function init_weights(size::Tuple{Int64, Int64})
    var = 2 / size[1]
    d = Normal(0, var)
    return convert.(Float32, rand(d, size))
end

function init_bias(size::Int64)
    var = 2 / size
    d = Normal(0, var)
    return convert.(Float32, rand(d, size))
end

function create(dimension::Int64, hidden_layer_structure::Vector{Int64}, activation::Function, normalise_inputs::Bool, use_physics::Bool)
    
    layers   = [2*dimension, hidden_layer_structure..., 1]
    num_maps = length(layers) -1
    
    weights = AutoGrad.Param[]
    biases  = AutoGrad.Param[]

    all_dims = Tuple{Int64, Int64}[]
    for i in 1:num_maps
        input_dim  = layers[i]
        output_dim = layers[i+1]
        
        push!(all_dims, (input_dim, output_dim))

    end
    
    weights = [AutoGrad.Param(init_weights(reverse(dims))) for dims in all_dims]
    biases  = [AutoGrad.Param(init_bias(dims[2])) for dims in all_dims]
    
    return HamiltonianNN(num_maps, weights, biases, activation, normalise_inputs,use_physics,  zeros(Float32, 4), zeros(Float32, 4))
        
end

function create(dimension::Int64, hidden_layer_structure::Vector{Int64}, normalise_inputs::Bool, use_physics::Bool)
    return create(dimension, hidden_layer_structure, α, normalise_inputs, use_physics)       
end
    

function Base.show(io::IO, network::HamiltonianNN)
    representation = "Hamiltonian Neural Network (Float32) with structure: 4"
    for bias in network.biases
        representation *= " → "
        representation *= string(length(bias))
    end
    
    print(io, representation)
end

Input = Union{Vector{Float32}, 
              Param{Vector{Float32}},
              AutoGrad.Result{Vector{Float32}}
              }
        

# The forward pass of the network
function linear(weights::Param{Matrix{Float32}}, bias::Param{Vector{Float32}}, input::Input)
    return weights * input .+ bias
end


function get_q(input::Input)
    return [1 0  0 0 ;
            0 1  0 0 ;
            0 0  0 0 ;
            0 0  0 0 ] * input
end


function get_p(input::Input)
    return [0 0  0 0 ;
            0 0  0 0 ;
            0 0  1 0 ;
            0 0  0 1 ] * input
end


function forward(network::HamiltonianNN, input::Input)
    norm_p = norm(get_p(input))

    # Normalise the input to be between [-1, 1]
    if network.normalise_inputs
        input = 2*(input .- network.training_min)
        input = input ./ (network.training_max .- network.training_min)
        input = input .- 1

        input = 10 * input
    end
    
    activated = input[:]

    for i in 1:(network.num_maps - 1)
        @inbounds output = linear(network.weights[i], network.biases[i], activated)
        activated = network.α.(output)
    end
    
    @inbounds activated = linear(network.weights[end], network.biases[end], activated)

    # Return the speed of sound times the size of momentum: c||p||
    if network.use_physics
        return norm(sum(activated)) * norm_p
    else
        return norm(sum(activate))
    end
end

# Make the network callable.
(network::HamiltonianNN)(input::Input) = forward(network, input)


function compute∇H(network::HamiltonianNN, input::Vector{Float32})
    x = AutoGrad.Param(input)
    output = @diff network(x)
    
    gradient = AutoGrad.full(AutoGrad.grad(output, x))
        
    return gradient
end

function compute∇H(network::HamiltonianNN, input::AutoGrad.Result{Vector{Float32}})
    output = @diff network(input)
    
    gradient = AutoGrad.full(AutoGrad.grad(output, input))
        
    return gradient
end

