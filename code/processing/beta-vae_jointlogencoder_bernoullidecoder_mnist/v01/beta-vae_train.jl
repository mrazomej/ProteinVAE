println("Loading packages...")

# Import project package
import AutoEncoderToolkit as AET

# Import ML libraries
import Flux

# Import library to load MNIST dataset
using MLDatasets: MNIST

# Import library to save models
import JLD2

# Import Random library for reproducibility
import Random

# Set seed for reproducibility
Random.seed!(42)

## =============================================================================

# Define learning rate
η = 1e-3

# Define number of epochs
n_epoch = 500

# Define β value
β = 0.1

## =============================================================================

println("Loading MNIST dataset...")

# Define number of samples in batch
n_batch = 64
# Define total number of data points
n_data = n_batch * 10
# Define total number of validation data points
n_val = n_batch * 2

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

# Define threshold for binarization
thresh = 0.5

# Binarize training data
train_data = Float32.(train_data .> thresh)

# Binarize validation data
val_data = Float32.(val_data .> thresh)

## =============================================================================

# Partition data into batches
train_loader = Flux.DataLoader(train_data, batchsize=n_batch, shuffle=true)

## =============================================================================

# Load model
vae = JLD2.load("./output/model.jld2")["model"]
# Load state
model_state = JLD2.load("./output/model.jld2")["model_state"]
# Set model state
Flux.loadmodel!(vae, model_state)

## =============================================================================

# Explicit setup of optimizer
opt_vae = Flux.Train.setup(
    Flux.Optimisers.Adam(η),
    vae
)

# Define loss functio kwargs
loss_kwargs = Dict(:β => β)

## =============================================================================

# Initialize arrays to save loss, entropy, and MSE
train_loss = Array{Float32}(undef, n_epoch)
val_loss = Array{Float32}(undef, n_epoch)
train_entropy = Array{Float32}(undef, n_epoch)
val_entropy = Array{Float32}(undef, n_epoch)
train_mse = Array{Float32}(undef, n_epoch)
val_mse = Array{Float32}(undef, n_epoch)

# Loop through epochs
for epoch in 1:n_epoch
    println("Epoch: $(epoch)\n")
    # Loop through batches
    for (i, x) in enumerate(train_loader)
        println("Epoch: $(epoch) | Batch: $(i) / $(length(train_loader))")
        # Train RHVAE
        AET.VAEs.train!(vae, x, opt_vae; loss_kwargs=loss_kwargs)
    end # for train_loader

    # Compute loss in training data
    train_loss[epoch] = AET.VAEs.loss(vae, train_data; β=β)
    # Compute loss in validation data
    val_loss[epoch] = AET.VAEs.loss(vae, val_data; β=β)

    # Forward pass training data
    train_outputs = vae(train_data)
    # Compute cross-entropy
    train_entropy[epoch] = Flux.Losses.logitbinarycrossentropy(
        train_outputs.p, train_data
    )
    # Compute MSE for training data
    train_mse[epoch] = Flux.mse(train_outputs.p, train_data)

    # Forward pass training data
    val_outputs = vae(val_data)
    # Compute cross-entropy
    val_entropy[epoch] = Flux.Losses.logitbinarycrossentropy(
        val_outputs.p, val_data
    )
    # Compute MSE for validation data
    val_mse[epoch] = Flux.mse(val_outputs.p, val_data)

    println("Epoch: $(epoch) / $(n_epoch)\n " *
            "- train_mse: $(train_mse[epoch])\n " *
            "- val_mse: $(val_mse[epoch])\n " *
            "- train_loss: $(train_loss[epoch])\n " *
            "- val_loss: $(val_loss[epoch])\n " *
            "- train_entropy: $(train_entropy[epoch])\n " *
            "- val_entropy: $(val_entropy[epoch])\n"
    )
end # for n_epoch

JLD2.jldsave(
    "./output/beta-vae_beta$(β)_epoch$(lpad(n_epoch, 4, "0")).jld2",
    model_state=Flux.state(vae),
    train_entropy=train_entropy,
    train_loss=train_loss,
    train_mse=train_mse,
    val_entropy=val_entropy,
    val_mse=val_mse,
    val_loss=val_loss,
)