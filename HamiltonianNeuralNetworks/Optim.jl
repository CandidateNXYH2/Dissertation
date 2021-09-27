using Random

function squared_norm(v::Input)
    return sum(v.^2)
end

function normalise(v::Array{Float32})
    len = norm(v)
    if len > 0
        return v ./ len
    end
    return v
end


function get_loss(network::HamiltonianNN, data::Matrix{Float32})
    num_samples = size(data)[2]
    loss = 0
    for i in shuffle(1:num_samples)
        # Get the rotated value of the Hamiltonian at that point
        @inbounds Ω∇H   = hamiltonian_vector_field(network, data[1:4, i])

        @inbounds loss += squared_norm(data[5:6, i]  - Ω∇H[1:2]) + squared_norm(data[7:8, i]  - Ω∇H[3:4])
    end
    
    return loss / num_samples
end

function get_weight_loss(network::HamiltonianNN)
    loss = 0
    for weight in network.weights 
        loss += sum(weight .^2 )
    end

    return loss
end

function get_gradient(network::HamiltonianNN, data::Matrix{Float32}, λ::Float32)
    loss     = @diff get_loss(network, data) + λ * get_weight_loss(network)
    gradient = [grad(loss, param) for param in params(network)]
    
    return get_loss(network, data), gradient 
end

function gradient_step(network::HamiltonianNN, data::Matrix{Float32}, 
                       η::Float32, μ::Float32, λ::Float32, parameter_velocities::Vector{Any},
                       threshold::Float32)   

#     Move the parameters for the network to the intermediate point
    @floop for (i,param) in enumerate(params(network))
        @inbounds value(param) .+=  μ*parameter_velocities[i]
    end

    # Get the gradient of the loss with respect to this network parameter
    loss, ∇params = get_gradient(network, data, λ)

    gradient = reduce(vcat, vec.(∇params))


    # Compute the look ahead gradient and take a step
    @floop for (i,param) in enumerate(params(network)) 
        if !isnothing(∇params[i])

            # Clip the gradient parameter-by-parameter rather than globally
            grad_norm = norm(∇params[i])
            if grad_norm > threshold
                # Normalise the gradient for the parameter
                ∇params[i] = threshold * ∇params[i] ./ grad_norm
            end

            # Take a step, with momentum
            @inbounds parameter_velocities[i]  = μ*parameter_velocities[i] .- η*∇params[i]
            @inbounds param .+= parameter_velocities[i]
        end
    end
    
    return loss
end
