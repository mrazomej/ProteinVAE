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

# Define number of samples in batch
n_batch = 64
# Define total number of data points
n_data = n_batch * 10

# Define RHVAE hyper-parameters
T = 0.4f0 # Temperature
λ = 1.0f-2 # Regularization parameter
n_centroids = 64 # Number of centroids

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

# Reduce size of data and reshape to WHCN format
data_train = Float32.(reshape(data_filt[:, :, 1:n_data], (28, 28, 1, n_data)))
labels_train = labels_filt[1:n_data]

## =============================================================================

# Select centroids via k-means
centroids_data = AutoEncode.utils.centroids_kmedoids(data_train, n_centroids)

## =============================================================================

println("Defining RHVAE model...")

# Define dimensionality of latent space
n_latent = 2

# Define number of initial channels
n_channels_init = 128

println("Defining encoder...")
# Define convolutional layers
conv_layers = Flux.Chain(
    # First convolutional layer
    Flux.Conv((3, 3), 1 => n_channels_init, Flux.relu; stride=2, pad=1),
    # Second convolutional layer
    Flux.Conv(
        (3, 3), n_channels_init => n_channels_init * 2, Flux.relu;
        stride=2, pad=1
    ),
    # Flatten the output
    AutoEncode.Flatten()
)

# Define layers for µ and log(σ)
µ_layer = Flux.Dense(n_channels_init * 2 * 7 * 7, n_latent, Flux.identity)
logσ_layer = Flux.Dense(n_channels_init * 2 * 7 * 7, n_latent, Flux.identity)

# build encoder
encoder = AutoEncode.JointLogEncoder(conv_layers, µ_layer, logσ_layer)

## =============================================================================

println("Defining decoder...")

# Define deconvolutional layers
deconv_layers = Flux.Chain(
    # Define linear layer out of latent space
    Flux.Dense(n_latent => n_channels_init * 2 * 7 * 7, Flux.identity),
    # Unflatten input using custom Reshape layer
    AutoEncode.Reshape(7, 7, n_channels_init * 2, :),
    # First transposed convolutional layer
    Flux.ConvTranspose(
        (4, 4), n_channels_init * 2 => n_channels_init, Flux.relu;
        stride=2, pad=1
    ),
    # Second transposed convolutional layer
    Flux.ConvTranspose(
        (4, 4), n_channels_init => 1, Flux.relu;
        stride=2, pad=1
    ),
    # Add normalization layer
    Flux.BatchNorm(1, Flux.sigmoid),
)

# Define decoder
decoder = AutoEncode.BernoulliDecoder(deconv_layers)


println("Combining encoder and decoder...")
vae = encoder * decoder
## =============================================================================

println("Defining Metric MLP...")

# Define convolutional layers
mlp_conv_layers = Flux.Chain(
    # Flatten the input using custom Flatten layer
    AutoEncode.Flatten(),
    # First layer
    Flux.Dense(28 * 28 => 400, Flux.relu),
    # Second layer
    Flux.Dense(400 => 400, Flux.relu),
    # Third layer
    Flux.Dense(400 => 400, Flux.relu),
)

# Define layers for the diagonal and lower triangular part of the covariance
# matrix
diag = Flux.Dense(400 => n_latent, Flux.identity)
lower = Flux.Dense(
    400 => n_latent * (n_latent - 1) ÷ 2, Flux.identity
)

# Build metric chain
metric_chain = AutoEncode.RHVAEs.MetricChain(mlp_conv_layers, diag, lower)

println("Building RHVAE model...")

rhvae = AutoEncode.RHVAEs.RHVAE(
    vae, metric_chain, centroids_data, T, λ
)

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => rhvae, "model_state" => Flux.state(rhvae))
)
