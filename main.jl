using AutoGrad
using BSON
using BSON: @save
using DelimitedFiles
using LinearAlgebra
using FLoops
using Printf
using Random
using UnicodePlots

include("HamiltonianNeuralNetworks/HamiltonianNeuralNetworks.jl")
using Main.HamiltonianNeuralNetworks 



function trainHNN(network::HamiltonianNN, training_steps::Int64, parameter_velocities::Vector{Any},
                  training_batch_size::Int64, testing_batch_size::Int64,   
                  η::Float32, μ::Float32, λ::Float32, threshold::Float32,
                  training_losses::Vector{Float32}, testing_losses::Vector{Float32},
                  savename::String)

    hyperparameters = [training_steps, training_batch_size, testing_batch_size, η, μ]

    best_H = deepcopy(network)

    # Load all the data
    data = readdlm("./expanded_dynamics.csv", ',', Float32)
    
    # Number of data samples: rays and their associated trajectories (i.e. dynamics time series data)
    num_samples = size(data)[2]

    # Determine the number of training samples
    test_train_split     = 0.9
    num_training_samples = convert(Int64, floor(test_train_split * num_samples))

    # Split the total 19,350 samples' indices into the testing and training split
    all_indices = shuffle(collect(1:num_samples))
    training_indices = all_indices[1:num_training_samples]
    testing_indices  = all_indices[num_training_samples:end]

    # Compute the values we need for normalisation
    minima = map(minimum, eachrow(data[1:4, training_indices]))
    maxima = map(maximum, eachrow(data[1:4, training_indices]))
    network.training_min = minima
    network.training_max = maxima

    saved_hamiltonians = HamiltonianNN[]

    min_test_loss = Inf
    print("\n\t\t\t\t======= Starting training =======\n")
    flush(stdout)

    for i in 1:training_steps
        "------------------------------Training--------------------------------------"
        # Generate the ray batches 
        batch_indices = rand(training_indices,  training_batch_size)
        batch         = data[:, batch_indices]

        num_completed = length(training_losses)
        should_print = (num_completed == 1) || (num_completed % 5 == 0)
        if should_print
            # Compute the loss and take a gradient step with respect to the batches we created above
            @time training_loss = gradient_step(network, batch, η, μ, λ, parameter_velocities, threshold)
        else 
            # Compute the loss and take a gradient step with respect to the batches we created above
            training_loss = gradient_step(network, batch, η, μ, λ, parameter_velocities, threshold)
        end

        push!(training_losses, training_loss)

        "------------------------------Testing---------------------------------------"
        # Generate the ray batches 
        testing_batch_indices = rand(testing_indices,   testing_batch_size)
        testing_batch         = data[:, testing_batch_indices]


        testing_loss = get_loss(network, testing_batch)
        push!(testing_losses,  testing_loss)


        # Save the current timestep if it's an improvement from the best we've seen
        if testing_loss < min_test_loss
           best_H = deepcopy(network)
           min_test_loss = testing_loss
        end

        @save savename best_H training_losses testing_losses parameter_velocities hyperparameters

        if should_print
            print(@sprintf("\n\nFinished step: %3d\n", num_completed))
            print("\tTraining loss: ", training_losses[end], "\n")
            print("\tTesting loss:  ",  testing_losses[end], "\n")
            flush(stdout)

            # Plot the results in the terminal
            p = lineplot( testing_losses,  color=:blue, name="Testing", title="Loss", canvas = DotCanvas)
            lineplot!(p, training_losses, color=:red,   name="Training")
            display(p)
        end

        should_save_timeseries = i % 25 == 0
        if should_save_timeseries
            push!(saved_hamiltonians, best_H)
            @save "saved_hamiltonians.bson" saved_hamiltonians
        end
    end
end


scale_inputs = true
use_physics  = true

if !scale_inputs && !use_physics
    savename = "savedata.bson"
elseif scale_inputs && !use_physics
    savename = "savedata_normalised.bson"
elseif !scale_inputs && use_physics
    savename = "savedata_helped.bson"
elseif scale_inputs && use_physics
    savename = "savedata_normalised_helped.bson"
end

H = create(2, [200,200], scale_inputs, use_physics)    
training_losses  = Float32[]
testing_losses   = Float32[]

# Initialise the parameter velocities for the momentum-based SGD
parameter_velocities = []
for param in params(H)
    push!(parameter_velocities, zeros(Float32, size(param)))
end


print(H)
training_steps   = 200000

train_batch_size = 2000
 test_batch_size = 1000


Δt = convert(Float32, 5e-8)
μ = convert(Float32, 0.9)  # Momentum
η = convert(Float32, 1e-3) # Learning rate
λ = convert(Float32, 0)
threshold = convert(Float32, 50)


trainHNN(H, training_steps, parameter_velocities,
         train_batch_size, test_batch_size, 
         η, μ, λ, threshold,
         training_losses, testing_losses,
         savename)
