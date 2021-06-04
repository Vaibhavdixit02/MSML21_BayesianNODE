#cd("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE")
#Pkg.activate(".")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC
using JLD, StatsPlots, Distributions, Random

Random.seed!(123)

u0 = [2.0; 0.0]
datasize = 40
tspan = (0.0, 1)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
mean_ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
ode_data = mean_ode_data .+ 0.1 .* randn(size(mean_ode_data)..., 30)

####DEFINE THE NEURAL ODE#####
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, relu),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

function l(θ)
    lp = logpdf(MvNormal(zeros(length(θ) - 1), θ[end]), θ[1:end-1])
    ll = -sum(abs2, ode_data .- predict_neuralode(θ[1:end-1]))
    return lp + ll
end
function dldθ(θ)
    x, lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
end

init = [Float64.(prob_neuralode.p); 1.0]
opt = DiffEqFlux.sciml_train(x -> -l(x), init, ADAM(0.05), maxiters = 1500)
pmin = opt.minimizer;

## -----------------------------
####### Perform inference


### Fit neural ode to the data
@model function fit_node(data)
    σ ~ InverseGamma(2, 3)
    p ~ MvNormal(pmin, 1.0)
    # Calculate predictions for the inputs given the params.
    predicted = predict_neuralode(p)
    # observe each prediction.

    ThreadsX.map(1:30) do i
        data[:,:,i] ~ MvNormal(predicted, σ)
    end
end

model = fit_node(ode_data); # fit model to average simulated data

function perform_inference(samplesize, pmin, num_chains)
    alg = SGHMC(; learning_rate = 0.01, momentum_decay = 0.1)
    chain = sample(model, alg, MCMCThreads(), samplesize, num_chains, progress=true);
    return chain
end

function map_loss(chain)
    chain_array = Array(chain)
    k = size(chain_array,1)
    losses = loss.([chain_array[i,:] for i in 1:k])
    return losses
end

callback = function (p, l, param; doplot = true)
  # plot current prediction against data
  display(l)
  plt = scatter(ode_data[1,:], ode_data[2,:], label = "data")
  sol = prob_node(u0, param);
  scatter!(plt, sol[1,:], sol[2,:], label = "prediction")
  if doplot
    display(plot(plt))
  end
  return false
end

# init at map point
using JLD
pinit = initial_params(dudt2);
opt = DiffEqFlux.sciml_train(loss, train_prob.p, ADAM(0.05), maxiters = 1500)
# opt = DiffEqFlux.sciml_train(loss, pmin_spiral, BFGS(), maxiters = 1500)

pmin = opt.minimizer;
save("pmin_spiral.jld", "pmin_spiral", pmin)

using JLD
pmin = load("pmin_spiral.jld")
pmin_spiral = pmin["pmin_spiral"]


sol = prob_node(u0, pmin_spiral);
plot()
display(scatter!(sol[1,:], sol[2,:]))
display(scatter!(ode_data[1,:], ode_data[2,:]))


function plot_chain(chain, losses)
    pl = plot()
    chain_array = Array(chain)
    len = size(chain_array,1)

    training_end = 1.0
    tei = 50  #training_end_idx

    scatter!(tsteps, ode_data[1,:], color = :red, label = "Data: Var1", title = "Spiral Neural ODE")
    scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Var2")
    plot!([training_end-0.0001,training_end+0.0001],[-2.2,1.3],lw=3,color=:green,label="Training Data End", linestyle = :dash)


    for k in 1:300
        resol = prob_node(u0, chain_array[rand(100:len), :])
        plot!(tsteps[1:tei], resol[1,:][1:tei], alpha=0.04, color = :red, label = "")
        plot!(tsteps[1:tei], resol[2,:][1:tei], alpha=0.04, color = :blue, label = "")
        plot!(tsteps[tei:end], resol[1,:][tei:end], alpha=0.04, color = :purple, label = "")
        plot!(tsteps[tei:end], resol[2,:][tei:end], alpha=0.04, color = :purple, label = "")
    end

    idx = findmin(losses)[2]
    prediction = prob_node(u0, chain_array[idx, :])
    plot!(tsteps, prediction[1,:], color=:black, w=2, label = "")
    plot!(tsteps, prediction[2,:], color=:black, w=2, label = "Training: Best fit prediction", ylims = (-2.5, 3.5))
    plot!(tsteps[tei:end], prediction[1,:][tei:end], color = :purple, w = 2, label = "")
    plot!(tsteps[tei:end], prediction[2,:][tei:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-2.5, 3.5))

    display(plot!([training_end-0.0001,training_end+0.0001],[-1,5],lw=3,color=:green,label="Training Data End", linestyle = :dash))


    ################## COUNTOUR PLOTS ###################################

    pl2 = scatter(ode_data[1,:], ode_data[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Spiral Neural ODE")

    for k in 1:300
        resol = prob_node(u0, chain_array[rand(50:len), :])
        plot!(resol[1,:][1:tei],resol[2,:][1:tei], alpha=0.04, color = :red, label = "")
        plot!(resol[1,:][tei:end],resol[2,:][tei:end], alpha=0.1, color = :purple, label = "")

    end

    plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Training: Best fit prediction", ylims = (-2.5, 3.5))
    display(plot!(prediction[1,:][tei:end], prediction[2,:][tei:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-2.5, 3.5)))

    return pl, pl2;
end


## ---------------------------------------------------
#
samples = 500

num_chains = 4;
chain = perform_inference(samples, pmin, num_chains)
for i in 1:num_chains
    losses = map_loss(chain[:,:,i])
    pl = plot(1:samples, losses); display(pl)
    savefig(pl, string("spiral_", lr, "_", md, "_", samples, "_", "chain_", i+4, "_losses", ".png"))
    pl_ch, pl2 = plot_chain(chain[:,:,i], losses)
    savefig(pl_ch, string("spiral_", lr, "_", md, "_", samples, "_", "chain_", i+4, "_predictions", ".png"))
    savefig(pl2, string("spiral_", lr, "_", md, "_", samples, "_", "chain_", i+4, "_contour", ".png"))
end

rand(MvNormal(pmin_spiral, 0.1))