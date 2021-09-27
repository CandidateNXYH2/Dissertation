function hamiltonian_vector_field(H::HamiltonianNN, coordinate::Input)
    Ω = [0 0    1 0 ;
         0 0    0 1 ;
        
        -1 0   0 0 ;
         0 -1   0 0 ]

    return Ω * compute∇H(H, coordinate)
end


function RK4_step(f::Function, h::Float32, coordinate::Input)
    k1 = f(coordinate)
    k2 = f(coordinate .+ 0.5f0 * h * k1)
    k3 = f(coordinate .+ 0.5f0 * h * k2)
    k4 = f(coordinate .+         h * k3)
   	
    return coordinate + (k1 + 2*k2 + 2*k3 + k4) * h / 6
end


function trace_step(network::HamiltonianNN, initial::Input, timestep::Float32)
    f = x -> hamiltonian_vector_field(network, x)
    return RK4_step(f, timestep, initial)
end