module VDPTag2Experiments

import ARDESPOT

using POMDPs
using MCTS
using ParticleFilters
using StaticArrays
using VDPTag2

using POMDPModelTools: GenerativeBeliefMDP
using Random: Random, MersenneTwister, AbstractRNG, randperm

export
    RootToNextMLFirst,
    ObsAdaptiveParticleFilter,
    VDPUpper

struct RootToNextMLFirst
    rng::MersenneTwister
end

function MCTS.next_action(gen::RootToNextMLFirst, p::VDPTagPOMDP, b, node)
    if isroot(node) && n_children(node) < 1
        target_sum=MVector(0.0, 0.0)
        agent_sum=MVector(0.0, 0.0)
        for s in particles(b::ParticleCollection)
            target_sum += s.target
            agent_sum += s.agent
        end
        next = VDPTag2.next_ml_target(mdp(p), target_sum/n_particles(b))
        diff = next-agent_sum/n_particles(b)
        return TagAction(false, atan(diff[2], diff[1]))
    else
        return rand(gen.rng, actions(p))
    end
end

MCTS.next_action(gen::RootToNextMLFirst, p::GenerativeBeliefMDP, b, node) = next_action(gen, p.pomdp, b, node)

struct ObsAdaptiveParticleFilter{P<:POMDP,S,R,RNG<:AbstractRNG} <: Updater
    pomdp::P
    resample::R
    max_frac_replaced::Float64
    n_init::Int
    rng::RNG
    _pm::Vector{S}
    _wm::Vector{Float64}
end

function ObsAdaptiveParticleFilter(p::POMDP, resample, max_frac_replaced, n_init, rng::AbstractRNG)
    S = statetype(p)
    return ObsAdaptiveParticleFilter(p, resample, max_frac_replaced, n_init, rng, S[], Float64[])
end

POMDPs.initialize_belief(up::ObsAdaptiveParticleFilter, d::Any) = ParticleCollection([rand(up.rng, d) for i in 1:up.n_init])
POMDPs.update(up::ObsAdaptiveParticleFilter, b, a, o) = update(up, resample(up.resample, b, up.rng), a, o)

function POMDPs.update(up::ObsAdaptiveParticleFilter, b::ParticleCollection, a, o)
    if n_particles(b) > 2*up.resample.n
        b = resample(up.resample, b, up.rng)
    end

    ps = particles(b)
    pm = up._pm
    wm = up._wm
    resize!(pm, 0)
    resize!(wm, 0)

    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(up.pomdp, s)
            all_terminal = false
            sp = @gen(:sp)(up.pomdp, s, a, up.rng)
            push!(pm, sp)
            od = observation(up.pomdp, s, a, sp)
            push!(wm, pdf(od, o))
        end
    end
    ws = sum(wm)
    if all_terminal || ws < eps(1.0/length(wm))
        # warn("All states in particle collection were terminal.")
        return initialize_belief(up, reset_distribution(up.pomdp, b, a, o))
    end

    pc = resample(up.resample, WeightedParticleBelief{statetype(up.pomdp)}(pm, wm, ws, nothing), up.rng)
    ps = particles(pc)

    mpw = max_possible_weight(up.pomdp, a, o)
    frac_replaced = up.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/mpw)
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(up.rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = new_particle(up.pomdp, b, a, o, up.rng)
    end
    return pc
end

max_possible_weight(pomdp::VDPTagPOMDP, a::TagAction, o) = 0.0
new_particle(pomdp::VDPTagPOMDP, a::TagAction, o) = error("shouldn't get here")
reset_distribution(p::POMDP, b, a, o) = initialstate(p)

struct VDPUpper end

function ARDESPOT.ubound(ub::VDPUpper, pomdp::POMDP, b::ARDESPOT.ScenarioBelief)
    if all(isterminal(pomdp, s) for s in particles(b))
        return 0.0
    else
        return mdp(cproblem(pomdp)).tag_reward
    end
end

Random.seed!(p::Policy, ::Any) = nothing

end
