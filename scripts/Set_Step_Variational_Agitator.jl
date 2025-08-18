# Packages
using Dice, Graphs, SimpleWeightedGraphs


# Set generating function

"""This creates a set with lenght n such that the first n-1 entries are 
randomized from [-10,10] and the last entry is the  sum of the other n-1 entries.
The intended solution for this set is the n-1 entries in one partition while the last entry is in another. But there may be a set 
where there are multiple entries and thus solutions."""

function sum_problem(set_length::Int) ::Vector{Int}
    set = []
    for i in 1:set_length-1
        k = 0
        while k == 0
            k = rand(-10:10)
        end
        push!(set,k)
    end
    push!(set, sum(set))
    return set
end



# Creating the computing function, this will be run to supply a single iteration of data.
# It is simply a modified version of the Agitator from Agitator_Experiments.ipnyb
# One thing to note is that convergence could still happen on the final attempt, 
# so the final value in the counts_list array is NOT an accurate representation.

function Set_Step_Variational_Agitator(set::Vector{Int}, time_total::Int, steps::Int, agitation_count::Int, samples::Int) ::Vector
    S = set
    # Make Graph
    graph_set = SimpleWeightedGraph(length(S))
    
    # Adding edges
    for i in range(1,length(S)-1)
        for j in range(i+1,length(S))
        # We iterate over all possible options, excluding duplicates to avoid weird issues. Thus the range(i,4). 
        # Then we check if i == j, then the edge doesn't exist. Otherwise, continue
            add_edge!(graph_set,i,j,(S[i]*S[j]))
        end
    end
    
    # Making model
    # Stop time in model units
    total_time = time_total
    # Number of timesteps
    num_steps = steps
    # Step size
    delta_t = total_time/num_steps
    # Makes model
    model = Dice.Model(graph_set, Dice.model_2_hybrid_coupling, delta_t)

    # Making randomized initial state
    num_vertices = Graphs.nv(model.graph)
    
    converged = 0
    diverged = 0
    counts_list = zeros(agitation_count)
    times = []
    # We assume set_up takes minimal time, bulk of computational time is solving max_cut
    # We take this time amount and divide by the number of samples to get avg convergence time
    time = @elapsed begin
    for _ in 1:samples
        state::Dice.Hybrid = Dice.get_random_hybrid(num_vertices, 2.0)
        agnum = 0
        for _ in 1:agitation_count
            agnum += 1
            state = Dice.propagate(model, num_steps, state)
            if sum(S .* state[1]) == 0
                converged += 1
                break
            end
            state = (state[1], Dice.get_random_cube(num_vertices, 2.0))
        end
        counts_list[agnum] += 1
    end
    end
    converge_ratio = converged/samples
    return [counts_list,converge_ratio,time]
end


# Now creating the thing to run

params = Dict{String, Any}(
    "set" => [4, -9, -8, -1, -6, -9, 1, -2, -10, -40], # Generate a set, or replace with custom
    "sim_time" => 6,
    "steps" => 1500,
    "samples" => 10000,
    "max_agitations" => 50 #How many agitations we will run, starting at 1 with the first continous shuffle to make initial state.
)

@show params

agiation_num = []
convergence_ratios = []
run_time = []
counts_lists = []

@time begin

    # Run all the simulations
    for agitation in 1:params["max_agitations"]
        
        result = Set_Step_Variational_Agitator(params["set"], params["sim_time"], params["steps"], agitation, params["samples"])
        
        push!(agiation_num, agitation)
        push!(counts_lists, result[1])
        push!(convergence_ratios, result[2])
        push!(run_time, result[3])
        
        println("The $(agitation)th iteration done, $(params["max_agitations"] - agitation) left")
    end

    out_name = "$(params["steps"]) Steps and $(params["max_agitations"]) Max Agitations"
    # Initialize the record file:
    open(out_name, "a") do outf
        println(outf, "#PARAMETERS")
        for param in keys(params)
            println(outf, "# $param = $(params[param])")
        end
        println(outf, "\n#Note, format for DATA is [agitation_num, convergence_ratio, time for num of samples in params, counts_list]")
        println(outf, "\n#DATA")
        println(outf, " counts = [")

        #Writing data to file
        for (agitation, counts, converge_ratio, times) in zip(agiation_num, counts_lists, convergence_ratios, run_time)
            println(outf, "\t[$agitation, $converge_ratio, $times, $counts],")
        end
        
        println(outf, "]\n")
    end

    println("The results are saved to $(out_name)")
end
