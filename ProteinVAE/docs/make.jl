using Documenter
using ProteinVAE

makedocs(
    sitename = "ProteinVAE",
    format = Documenter.HTML(),
    modules = [ProteinVAE]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
