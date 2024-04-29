## =============================================================================
println("Loading packages...")

# Import project package
import ProteinVAE as PV

# Import project package
import AutoEncode

# Import libraries to handel data
import XLSX
import CSV
import DataFrames as DF
import Glob

# Import ML libraries
import Flux
import CUDA

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 5_000
# Define number of samples in batch
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rate
η = 10^-3

# Define loss function hyper-parameters
ϵ = Float32(1E-4) # leapfrog step size
K = 5 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define ELBO prefactors
logp_prefactor = [10.0f0, 1.0f0, 1.0f0]
logq_prefactor = [1.0f0, 1.0f0, 1.0f0]

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
)

# Define loss function kwargs
loss_kwargs = Dict(
    :K => K,
    :ϵ => ϵ,
    :βₒ => βₒ,
    :logp_prefactor => logp_prefactor,
    :logq_prefactor => logq_prefactor,
)
## =============================================================================

println("Loading data into memory...")

println("Load data...\n")

# List data directory
data_dir = "$(git_root())/data/lian_2023"

# List data files
files = sort(Glob.glob("$(data_dir)/Library*.xlsx"[2:end], "/"))

# Read excel file into DataFrame
df = XLSX.readtable(last(files), "Sheet1") |> DF.DataFrame

# Rename all columns to lowercase
DF.rename!(df, Dict(zip(DF.names(df), lowercase.(DF.names(df)))))

## =============================================================================

println("One-hot encode sequences...\n")

# One-hot encode sequences
seq_onehot = PV.seq.seq_onehotbatch(String.(df.sequence_aligned))

# Convert to Float32
seq_onehot = Float32.(Array(seq_onehot)) |> Flux.gpu

## =============================================================================

println("Preparing Data...")

# Assuming `ic50_std` is your data
train_data, val_data = Flux.splitobs(seq_onehot, at=split_frac, shuffle=true)
# Convert to CuArray
train_data = CUDA.CuArray(train_data)
val_data = CUDA.CuArray(val_data)

train_loader = Flux.DataLoader(train_data, batchsize=n_batch, shuffle=false)

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/model_state/"

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

## ============================================================================= 

println("Load model...\n")

# Load model
rhvae = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae, model_state)
# Update metric parameters
AutoEncode.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(size(rhvae.vae.encoder.µ.weight, 1))`
## Number of epochs
`n_epoch = $(n_epoch)`
## Training regimen
`rhvae mini-batch training on GPU`
## leapfrog step size
`ϵ = $(ϵ)
## Batch size
`n_batch = $(n_batch)`
## Data split fraction
`split_frac = $(split_frac)`
## Optimizer
`Adam($η)`
"""

# Write README file into memory
open("./output/model_state/README.md", "w") do file
    write(file, readme)
end

## ============================================================================= 

# List previous model parameters
model_states = sort(Glob.glob("$(out_dir)/beta-rhvae_epoch*.jld2"))

# Check if model states exist
if length(model_states) > 0
    # Load model state
    model_state = JLD2.load(model_states[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(rhvae, model_state)
    # Update metric parameters
    AutoEncode.RHVAEs.update_metric!(rhvae)
    # Extract epoch number
    epoch_init = parse(
        Int, match(r"epoch(\d+)", model_states[end]).captures[1]
    )
else
    epoch_init = 1
end # if

## =========================================================================

# Upload model to GPU
rhvae = Flux.gpu(rhvae)

# Explicit setup of optimizer
opt_rhvae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    rhvae
)

## =========================================================================

println("\nTraining RHVAE...\n")

# Loop through number of epochs
for epoch in epoch_init:n_epoch
    # Loop through batches
    for (i, x) in enumerate(train_loader)
        println("Epoch: $(epoch) | Batch: $(i) / $(length(train_loader))")
        # Train RHVAE
        AutoEncode.RHVAEs.train!(
            rhvae, x, opt_rhvae; loss_kwargs=loss_kwargs
        )
    end # for train_loader

    # Compute loss in training data
    loss_train = AutoEncode.RHVAEs.loss(
        rhvae, train_data; loss_kwargs...
    )
    # Compute loss in validation data
    loss_val = AutoEncode.RHVAEs.loss(
        rhvae, val_data; loss_kwargs...
    )

    # Forward pass training data
    local rhvae_train = rhvae(train_data; rhvae_kwargs..., latent=true)
    # Compute log-likelihood for training data
    local loglike_train = StatsBase.mean(
        AutoEncode.decoder_loglikelihood(
            train_data,
            rhvae_train.phase_space.z_init,
            rhvae.vae.decoder,
            rhvae_train.decoder
        )
    )

    # Forward pass validation data
    local rhvae_val = rhvae(val_data; rhvae_kwargs..., latent=true)
    # Compute MSE for validation data
    local loglike_val = StatsBase.mean(
        AutoEncode.decoder_loglikelihood(
            val_data,
            rhvae_train.phase_space.z_init,
            rhvae.vae.decoder,
            rhvae_val.decoder
        )
    )

    println("\n Epoch: $(epoch) / $(n_epoch)\n - loglike_train: $(loglike_train)\n - loglike_val: $(loglike_val)\n - loss_train: $(loss_train)\n - loss_val: $(loss_val)")

    # Save checkpoint
    JLD2.jldsave(
        "$(out_dir)/beta-rhvae_epoch$(lpad(epoch, 5, "0")).jld2",
        model_state=Flux.state(rhvae)|> Flux.cpu,
        loglike_train=loglike_train,
        loglike_val=loglike_val,
        loss_train=loss_train,
        loss_val=loss_val
    )
end # for n_epoch