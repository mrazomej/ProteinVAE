## 
println("Loading packages...")

# Import project package
import AutoEncode

# Import ML libraries
import Flux

# Import library to load MNIST dataset
using MLDatasets: MNIST

# Import library to save models
import JLD2
# Import basic math
import Random
import LinearAlgebra
import StatsBase

Random.seed!(42)

## =============================================================================

# Define number of epochs
n_epoch = 100
# Define number of samples in batch
n_batch = 64
# Define total number of data points
n_data = n_batch * 10
# Define number of validation data points
n_val = n_batch * 2
# Define learning rate
η = 10^-3

# Define loss function hyper-parameters
ϵ = Float32(1E-4) # Leapfrog step size
K = 5 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define T values
T_vals = Float32.([0.2, 0.4, 0.6, 0.8])

# Define ELBO prefactors
logp_prefactor = [10.0f0, 1.0f0, 1.0f0]
logq_prefactor = [1.0f0, 1.0f0, 1.0f0]

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
)

# Define loss function hyper-parameters
loss_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
    :logp_prefactor => logp_prefactor,
    :logq_prefactor => logq_prefactor,
)

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/model_state"

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

## =============================================================================

println("Loading dataset...")

# Define lables to keep
digit_label = [0, 1, 2]

# Load data and labels
dataset = MNIST(;
    split=:train,
    dir="$(git_root())/data/mnist"
)

# Keep only data with labels in digit_label
data_filt = dataset.features[:, :, dataset.targets.∈Ref(digit_label)]
labels_filt = dataset.targets[dataset.targets.∈Ref(digit_label)]

# Reduce size of training data and reshape to WHCN format
train_data = Float32.(reshape(data_filt[:, :, 1:n_data], (28, 28, 1, n_data)))
train_labels = labels_filt[1:n_data]

# Reduce size of validation data and reshape to WHCN format
val_data = Float32.(
    reshape(data_filt[:, :, n_data+1:n_data+n_val], (28, 28, 1, n_val))
)
val_labels = labels_filt[n_data+1:n_data+n_val]

## =============================================================================

println("Preparing DataLoader...")

train_loader = Flux.DataLoader(train_data, batchsize=n_batch, shuffle=true)

# Upload to GPU
train_data = Flux.gpu(train_data)
val_data = Flux.gpu(val_data)

# Extract each batch
train_batches = Flux.gpu.([x for x in train_loader])

## =============================================================================

println("Loading RHVAE...")

# Load model 
rhvae_template = JLD2.load("./output/model.jld2")["model"]
# Load state
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Set model state
Flux.loadmodel!(rhvae_template, model_state)

## =============================================================================

println("Training RHVAE...\n")

# Loop through β values
for i in eachindex(T_vals)
    # Define λ value
    T = T_vals[i]
    # Make a copy of the model   
    rhvae = AutoEncode.RHVAEs.RHVAE(
        deepcopy(rhvae_template.vae),
        deepcopy(rhvae_template.metric_chain),
        deepcopy(rhvae_template.centroids_data),
        deepcopy(rhvae_template.centroids_latent),
        deepcopy(rhvae_template.L),
        deepcopy(rhvae_template.M),
        T,
        deepcopy(rhvae_template.λ)
    ) |> Flux.gpu

    # Update metric parameters
    AutoEncode.RHVAEs.update_metric!(rhvae)

    # Explicit setup of optimizer
    opt_rhvae = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        rhvae
    )
    # Loop through number of epochs
    for epoch in 1:n_epoch
        println("\n($T) | Epoch: $(epoch)\n")
        # Loop through batches
        for (i, x) in enumerate(train_batches)
            println("$(T) | Epoch: $(epoch) | Batch: $(i) / $(length(train_loader))")
            # Train RHVAE
            AutoEncode.RHVAEs.train!(
                rhvae, x, opt_rhvae; loss_kwargs=loss_kwargs, verbose=false
            )
        end # for train_loader

        # Compute loss in training data
        train_loss = AutoEncode.RHVAEs.loss(
            rhvae, train_data; loss_kwargs...
        )
        # Compute loss in validation data
        val_loss = AutoEncode.RHVAEs.loss(
            rhvae, val_data; loss_kwargs...
        )

        # Forward pass training data
        local train_outputs = rhvae(train_data; rhvae_kwargs...)
        # Compute cross-entropy
        local train_entropy = Flux.Losses.logitbinarycrossentropy(
            train_outputs.p, train_data
        )
        # Compute MSE for training data
        local train_mse = Flux.mse(train_outputs.p, train_data)

        # Forward pass training data
        local val_outputs = rhvae(val_data; rhvae_kwargs...)
        # Compute cross-entropy
        local val_entropy = Flux.Losses.logitbinarycrossentropy(
            val_outputs.p, val_data
        )
        # Compute MSE for validation data
        local val_mse = Flux.mse(val_outputs.p, val_data)

        println("\n ($(T)) Epoch: $(epoch) / $(n_epoch)\n - train_mse: $(train_mse)\n - val_mse: $(val_mse)\n - train_loss: $(train_loss)\n - val_loss: $(val_loss)\n - train_entropy: $(train_entropy)\n - val_entropy: $(val_entropy)\n")
        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/rhvae_$(T)temp_epoch$(lpad(epoch, 4, "0")).jld2",
            model_state=Flux.cpu(Flux.state(rhvae)),
            train_entropy=train_entropy,
            train_loss=train_loss,
            train_mse=train_mse,
            val_entropy=val_entropy,
            val_mse=val_mse,
            val_loss=val_loss,
        )
    end # for n_epoch
end # for T_vals