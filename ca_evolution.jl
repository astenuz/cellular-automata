using Evolutionary: initial_population
# Implementation of an elementary cellular automata
# according to https://mathworld.wolfram.com/ElementaryCellularAutomaton.html
# uses random initialization and uses wolfram codes to specify the rule
# synchronous update of the 1D lattice

using Agents, Random
using CairoMakie
using InteractiveDynamics
using CSV
using Distributions
using Evolutionary


RADIUS = 3
RULE_LENGTH = 2^(2*RADIUS+1)
N_CELLS = 149

"""
The automaton living in a 1D space
"""
mutable struct Cell <: AbstractAgent
    id::Int
    pos::Dims{1}
    status::Bool # either 0 or 1, where 1 is 'alive'
end

"""
Takes the status of a neighborhood and returns the corresponding 
index in the model rule (0,0,0) -> 1 (0,1,0) -> 3
"""
function configuration_index(cell_statuses)
    # takes the tuples and forms a string (0,1,1) -> "011"
    binary_code = string(UInt8.(cell_statuses)...)
    # turns the string binary code into the integer
    index = parse(Int, binary_code, base=2) + 1
    return index
end

"""
Given a cell checks its neighbors and 
decides the next status of the cell based on the model rule
"""
function next_status(cell, model)
    neighbors = collect(nearby_agents(cell, model, model.radius))
    cell_statuses = (c.status for c in vcat(neighbors[1:model.radius], cell, neighbors[model.radius+1:end]))
    index = configuration_index(cell_statuses)
    return model.rule[index]
end

"""
Initializes the ABM

# Arguments
- `n_cells::Int`: total of cell automaton
- `wolfram_code::Int`: wolfram code for the rule
- `seed::Int`: random seed
"""
function build_model(rule; n_cells = 100, cell_p=0.5, radius=3, seed = 30)
    space = GridSpace((n_cells,); metric=:chebyshev)

    properties = Dict(:rule => rule, :radius => radius)
    model = ABM(
        Cell, 
        space; 
        properties,
        rng = MersenneTwister(seed))

    dist = Bernoulli(cell_p)
    for x in 1:n_cells
        cell = Cell(nextid(model), (x,), rand(dist))
        add_agent_pos!(cell, model)
    end

    return model
end

"""
Dummy update of the cells
"""
function cell_step!(_, _)

end

"""
Performs a synchronous update of all cells
"""
function ca_step!(model)
    new_statuses = fill(false, nagents(model))
    for agent in allagents(model)
        new_statuses[agent.id] = next_status(agent, model)
    end

    for id in allids(model)
        model[id].status = new_statuses[id]
    end
end

allsame(x) = all(y -> y == first(x), x)

function predominant_status(model)
    statuses = (c.status for c in allagents(model))
    return mean(statuses) >= 0.5
end

function score_scenario(model, target)
    statuses = (c.status for c in allagents(model))
    return all(y -> y == target, statuses)
end


function evaluate_rule(rule; n_cells=149, n_scenarios=100, max_steps=2*n_cells)
    cell_ps = rand(Uniform(0,1), n_scenarios)
    results = falses(n_scenarios)

    for i in 1:n_scenarios
        model = build_model(rule; n_cells=n_cells, cell_p=cell_ps[i], radius=RADIUS)
        target = predominant_status(model)
        run!(model, cell_step!, ca_step!, max_steps);
        results[i] = score_scenario(model, target)
    end

    return 1 - mean(results) # the lower the better
end


function random_rule(p)
    dist = Bernoulli(p)
    return rand(dist, RULE_LENGTH) .== 1
end

#println("running test eval")
#rintln(evaluate_rule(random_rule(0.5)))


println("Running optimization")
POP_SIZE = 100
MAX_ITERATIONS = 100

initial_rules = [random_rule(p) for p in rand(Uniform(0,1), POP_SIZE)]

ga = GA(
    populationSize = POP_SIZE,
    mutationRate = 0.15,
    epsilon = 0.1,
    selection = susinv,
    crossover = twopoint,
    mutation = flip
)

population = Evolutionary.initial_population(ga, initial_rules)

#result = Evolutionary.optimize(
#    evaluate_rule,
#    population,
#    ga,
#    Evolutionary.Options(
#        iterations=MAX_ITERATIONS, store_trace=true, show_trace=true, parallelization=:thread))


function plot_rule(rule; n_cells=149, cell_p=0.5, max_steps=2*n_cells)
    model = build_model(rule; n_cells=n_cells, cell_p=cell_p, radius=RADIUS)
    data, _ = run!(model, cell_step!, ca_step!, max_steps; adata=[:status])

    fig, ax = heatmap(data.id, data.step, data.status, colormap=:Blues_3)
    ax.yreversed = true
    return fig
end

rule_decimals(rule_hex) = [parse(Int, c, base= 16) for c in  rule_hex]
rule_from_decimals(rule_decimals) = BitArray(vcat([reverse(digits(d, base=2, pad=4)) for d in rule_decimals]...))
rule_from_hex(rule_hex) = (rule_from_decimals ∘ rule_decimals)(rule_hex)

maj_rule = BitArray(mean(digits(i, base=2, pad=7)) .> 0.5 for i in 0:127)

par_rule_hex = "0504058705000f77037755837bffb77f"
par_rule = rule_from_hex(par_rule_hex)

#plot_rule(Evolutionary.minimizer(result))

# Initialize model
#model = build_model(random_rule(0.5); n_cells=149, cell_p=0.5, radius=3)

# Runs the model and collects data
#data, _ = run!(model, cell_step!, ca_step!, 100; adata=[:status]);

# The data contains the step, id/position and status of the cell (1/0)
#data

#CSV.write("ca_data.csv", data);

# Lets plot the time evolution in the y axis
#heatmap(data.id, data.step, data.status, colormap=:Blues_3)