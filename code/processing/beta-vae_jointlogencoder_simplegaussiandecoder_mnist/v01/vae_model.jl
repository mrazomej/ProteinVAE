println("Loading packages...")

# Import project package
import AutoEncoderToolkit as AET

# Import ML libraries
import Flux

# Import library to load MNIST dataset
using MLDatasets: MNIST

# Import library to save models
import JLD2

## =============================================================================

println("Loading MNIST dataset...")

# Define number of samples in batch
n_batch = 64
# Define total number of data points
n_data = n_batch * 10

# Define lables to keep
digit_label = [0, 1, 2]

# Load data and labels
data, labels = MNIST.traindata(
    ; dir="$(git_root())/data/mnist"
)

# Keep only data with labels in digit_label
data_filt = data[:, :, labels.∈Ref(digit_label)]
labels_filt = labels[labels.∈Ref(digit_label)]

# Reduce size of data and reshape to WHCN format
data_train = Float32.(reshape(data_filt[:, :, 1:n_data], (28, 28, 1, n_data)))
labels_train = labels_filt[1:n_data]

## =============================================================================

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
    AET.Flatten()
)

# Define layers for µ and log(σ)
µ_layer = Flux.Dense(n_channels_init * 2 * 7 * 7, n_latent, Flux.identity)
logσ_layer = Flux.Dense(n_channels_init * 2 * 7 * 7, n_latent, Flux.identity)

# build encoder
encoder = AET.JointGaussianLogEncoder(conv_layers, µ_layer, logσ_layer)

## =============================================================================

println("Defining decoder...")

# Define deconvolutional layers
deconv_layers = Flux.Chain(
    # Define linear layer out of latent space
    Flux.Dense(n_latent => n_channels_init * 2 * 7 * 7, Flux.identity),
    # Unflatten input using custom Reshape layer
    AET.Reshape(7, 7, n_channels_init * 2, :),
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
decoder = AET.SimpleGaussianDecoder(deconv_layers)

## =============================================================================

# Define VAE model
vae = encoder * decoder

## =============================================================================

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => vae, "model_state" => Flux.state(vae))
)