## =============================================================================
println("Import packages")

# Import project package
import ProteinVAE as PV

# Import library to list files
import Glob

# Import Dataframe-related libraries
import XLSX
import DataFrames as DF

# Import ML Libraries
import Flux
import AutoEncode

# Import Basic math
import Distances
import StatsBase

# Import library to save model
import JLD2

# Import Random
import Random

Random.seed!(42)
## =============================================================================

# Define model hyperparameters

# Define latent dimension
latent_dim = 2

# Define RHVAE hyper-parameters
T = 0.8f0 # Temperature
λ = 1.0f-2 # Regularization parameter
n_centroids = 128 # Number of centroids


## =============================================================================

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

# Extract sequence length, number of unique characters, and number of sequences
n_char, seq_len, n_seq = size(seq_onehot)

# Select centroids via k-means
centroids_data = AutoEncode.utils.centroids_kmedoids(
    seq_onehot, n_centroids, Distances.Hamming()
)

## =============================================================================

println("Define model...\n")

# Define number input features
n_input = n_char * seq_len
# Define expansion size
n_neuron = floor(Int64, n_input * 1.5)

println("Define encoder...\n")

# Define encoder mlp

encoder_mlp = Flux.Chain(
    # Flatten input with custom layer
    AutoEncode.Flatten(),
    # Encoder first layer
    Flux.Dense(n_input => n_neuron, Flux.identity),
    # Enconder second layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Enconder third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Encoder fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
)

# Define encoder µ and log(σ) layers
encoder_µ = Flux.Dense(n_neuron => latent_dim, Flux.identity)
encoder_logσ = Flux.Dense(n_neuron => latent_dim, Flux.identity)

# Build encoder
encoder = AutoEncode.JointLogEncoder(encoder_mlp, encoder_µ, encoder_logσ)

## =============================================================================

println("Define decoder...\n")

decoder = AutoEncode.CategoricalDecoder(
    # Define decoder mlp
    Flux.Chain(
        # Decoder first layer
        Flux.Dense(latent_dim => n_neuron, Flux.identity),

        # Decoder second layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Decoder third layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Decoder fourth layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),

        # Define output layer
        Flux.Dense(n_neuron => n_input, Flux.identity),
        # Reshape output with custom layer
        AutoEncode.Reshape(n_char, seq_len, :),

        # Apply softmax for each amino acid
        AutoEncode.ActivationOverDims(Flux.softmax, 1),
    )
)

## =============================================================================

println("Defining Metric MLP...\n")

# Define metric mlp
metric_mlp = Flux.Chain(
    # Flatten input using custom layer
    AutoEncode.Flatten(),
    # Metric first layer
    Flux.Dense(n_input => n_neuron, Flux.identity),
    # Metric second layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Metric third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Metric fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
)

# Define diag and lower layers
diag = Flux.Dense(n_neuron => latent_dim, Flux.identity)
lower = Flux.Dense(n_neuron => latent_dim * (latent_dim - 1) ÷ 2, Flux.identity)

metric_chain = AutoEncode.RHVAEs.MetricChain(metric_mlp, diag, lower)

## =============================================================================

println("Define RHVAE...\n")

rhvae = AutoEncode.RHVAEs.RHVAE(
    encoder * decoder, metric_chain, centroids_data, T, λ
)

## =============================================================================

println("Setting output directories...")

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

## =============================================================================

println("Saving model...")

JLD2.save(
    "./output/model.jld2",
    Dict("model" => rhvae, "model_state" => Flux.state(rhvae))
)
