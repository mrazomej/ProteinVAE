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

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define model hyperparameters

# Define number of epochs
n_epoch = 1_000
# Define number of samples in batch
n_batch = 256
# Define fraction to split data into training and validation
split_frac = 0.85
# Define learning rate
η = 10^-3

# Define β values to train
ϵ_vals = Float32.([1E-4, 1E-3, 1E-2, 1E-1])

# Define loss function hyper-parameters
K = 5 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

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
seq_onehot = Float32.(Array(seq_onehot))

## =============================================================================

println("Preparing Data...")

# Assuming `ic50_std` is your data
train_data, val_data = Flux.splitobs(seq_onehot, at=split_frac, shuffle=true)

train_loader = Flux.DataLoader(train_data, batchsize=n_batch, shuffle=true)

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
rhvae_template = JLD2.load("./output/model.jld2")["model"]
# Load parameters
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Input parameters to model
Flux.loadmodel!(rhvae_template, model_state)
# Update metric parameters
AutoEncode.RHVAEs.update_metric!(rhvae_template)

# Make a copy of the model for each β value
models = [Flux.deepcopy(rhvae_template) for _ in ϵ_vals]

## =============================================================================

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(size(rhvae_template.vae.encoder.µ.weight, 1))`
## Number of epochs
`n_epoch = $(n_epoch)`
## Training regimen
`rhvae mini-batch training witm multiple ϵ (leapfrog step size) values`
## ϵ values
`ϵ_vals = $(ϵ_vals)
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

# Loop through β values
Threads.@threads for i in eachindex(ϵ_vals)
    # Define model
    rhvae = models[i]
    # Define ϵ value
    ϵ = ϵ_vals[i]

    # Define RHVAE hyper-parameters in a dictionary
    rhvae_kwargs = Dict(
        :K => K,
        :ϵ => ϵ,
        :βₒ => βₒ,
        :∇H => AutoEncode.RHVAEs.∇hamiltonian_TaylorDiff,
    )

    # List previous model parameters
    model_states = sort(Glob.glob("$(out_dir)/rhvae_$(ϵ)epsilon_epoch*.jld2"))

    # Check if model states exist
    if length(model_states) > 0
        # Load model state
        model_state = JLD2.load(model_states[end])["model_state"]
        # Input parameters to model
        Flux.loadmodel!(rhvae, model_state)
        # Update metric parameters
        Flux.update_metric!(rhvae)
        # Extract epoch number
        epoch_init = parse(
            Int, match(r"epoch(\d+)", model_states[end]).captures[1]
        )
    else
        epoch_init = 1
    end # if

    ## =========================================================================

    # Explicit setup of optimizer
    opt_rhvae = Flux.Train.setup(
        Flux.Optimisers.Adam(η),
        rhvae
    )

    ## =========================================================================

    println("Training RHVAE...\n")

    # Loop through number of epochs
    for epoch in epoch_init:n_epoch
        # Loop through batches
        for (i, x) in enumerate(train_loader)
            # Train RHVAE
            AutoEncode.RHVAEs.train!(
                rhvae, x, opt_rhvae; loss_kwargs=rhvae_kwargs
            )
        end # for train_loader

        # Compute loss in training data
        loss_train = AutoEncode.RHVAEs.loss(
            rhvae, train_data; rhvae_kwargs...
        )
        # Compute loss in validation data
        loss_val = AutoEncode.RHVAEs.loss(
            rhvae, val_data; rhvae_kwargs...
        )

        # Forward pass training data
        local rhvae_train = rhvae(train_data; rhvae_kwargs...)
        # Compute log-likelihood for training data
        local loglike_train = AutoEncode.decoder_loglikelihood(
            train_data,
            rhvae_train.phase_space.z_init,
            rhvae.vae.decoder,
            rhvae_train.decoder
        )

        # Forward pass validation data
        local rhvae_val = rhvae(val_data; rhvae_kwargs...)
        # Compute MSE for validation data
        local loglike_val = AutoEncode.decoder_loglikelihood(
            val_data,
            rhvae_train.phase_space.z_init,
            rhvae.vae.decoder,
            rhvae_val.decoder
        )


        println("\n ($(ϵ)) Epoch: $(epoch) / $(n_epoch)\n - loglike_train: $(loglike_train)\n - loglike_val: $(loglike_val)\n - loss_train: $(loss_train)\n - loss_val: $(loss_val)")

        # Save checkpoint
        JLD2.jldsave(
            "$(out_dir)/rhvae_$(ϵ)epsilon_epoch$(lpad(epoch, 5, "0")).jld2",
            model_state=Flux.state(rhvae),
            loglike_train=loglike_train,
            loglike_val=loglike_val,
            loss_train=loss_train,
            loss_val=loss_val
        )
    end # for n_epoch
end # for ϵ_vals
