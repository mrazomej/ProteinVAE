println("Loading packages...")

# Load project package
import ProteinVAE

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

# Import plotting library
using CairoMakie
CairoMakie.activate!()

# Set plotting style
ProteinVAE.viz.theme_makie!()

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

# Initialize figure
fig = Figure(size=(600, 400))

# Add axis
axes = [
    Axis(fig[i, j], aspect=AxisAspect(1), yreversed=true)
    for i in 1:2, j in 1:3
]

Random.seed!(42)
# Extract index of each digit
idx_digit = [findall(x -> x == d, train_labels) for d in digit_label]

# Extract two random indices for each digit
idx_plot = vcat([rand(idx_digit[d], 2) for d in 1:length(digit_label)]...)

# Loop through axes
for (i, ax) in enumerate(axes)
    # Add images to figure
    heatmap!(
        ax, train_data[:, :, 1, idx_plot[i]], show_axis=false, colormap=:grays,
    )
    # Turn off decorations
    hidedecorations!(ax)
end # for

# Save figure
# save("/Users/mrazo/git/AutoEncoderToolkit/docs/src/figs/bin_mnist.svg", fig)

fig

## =============================================================================

# Load model
vae = JLD2.load("./output/model.jld2")["model"]

# Define number of epochs
n_epoch = 500

# Load training results
train_results = JLD2.load(
    "./output/beta-vae_beta0.1_epoch$(lpad(n_epoch, 4, "0")).jld2"
)

## =============================================================================

# Plot loss and mse

# Initialize figure
fig = Figure(size=(750, 250))

# Add loss axis
axloss = Axis(
    fig[1, 1],
    xlabel="epoch",
    ylabel="VAE loss",
    title="Loss vs Epoch"
)

# Plot loss
lines!(axloss, 1:n_epoch, train_results["train_loss"], label="train set")
lines!(axloss, 1:n_epoch, train_results["val_loss"], label="validation set")

# Add legend
axislegend(axloss)

# Add axis for cross-entropy
axentropy = Axis(
    fig[1, 2],
    xlabel="epoch",
    ylabel="cross-entropy",
    title="Cross-Entropy vs Epoch"
)

# Plot cross-entropy
lines!(
    axentropy, 1:n_epoch, train_results["train_entropy"], label="train set"
)
lines!(
    axentropy, 1:n_epoch, train_results["val_entropy"], label="validation set"
)

# Add mse axis
axmse = Axis(
    fig[1, 3],
    xlabel="epoch",
    ylabel="mean squared error",
    title="MSE vs Epoch"
)

# Plot loss
lines!(axmse, 1:n_epoch, train_results["train_mse"], label="train set")
lines!(axmse, 1:n_epoch, train_results["val_mse"], label="validation set")

# Save figure
# save("/Users/mrazo/git/AutoEncoderToolkit/docs/src/figs/vae_train.svg", fig)

fig

## =============================================================================

# Map training data to latent space
train_latent = vae.encoder(train_data).µ

## =============================================================================

# Plot latent space

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="Latent Space"
)

# Loop through labels
for l in digit_label
    # Get indices of data with label l
    idx = findall(train_labels .== l)
    # Plot data
    scatter!(
        ax,
        train_latent[1, idx],
        train_latent[2, idx],
        label="digit $(l)",
        marker=first("$(l)")
    )
end

# Save figure
# save("/Users/mrazo/git/AutoEncoderToolkit/docs/src/figs/vae_latent.svg", fig)

fig

## =============================================================================

# Define latent space dimensionality
n_latent = 2
# Define number of samples
n_samples = 6

# Sample from prior
Random.seed!(42)
prior_samples = Random.randn(n_latent, n_samples)

# Decode samples
decoder_output = vae.decoder(prior_samples).p