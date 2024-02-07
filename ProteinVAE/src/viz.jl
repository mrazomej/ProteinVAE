# Import plotting-related libraries
using Measures, CairoMakie
import Makie
import ColorSchemes
import ColorTypes
import Colors


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Color Style
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@doc raw"""
    `theme_makie()`

Set plotting default to personal style for the `makie` plotting library. This
can be for either the GLMakie or the CairoMakie backends.
"""
function theme_makie!()
    # Seaborn colorblind
    colors = ColorSchemes.seaborn_colorblind

    theme = Theme(
        fonts=(;
            regular="Roboto Light",
            bold="Roboto Regular",
            italic="Roboto Light Italic",
            bold_italic="Roboto Regular Italic",
            extra_bold="Roboto Bold",
            extra_bold_italic="Roboto Bold Italic"
        ),
        Figure=(
            resolution = (300, 300)
        ),
        Axis=(
            # backgroundcolor="#EAEAF2", 
            backgroundcolor="#E6E6EF",

            # Font sizes
            titlesize=16,
            xlabelsize=16,
            ylabelsize=16,
            xticklabelsize=14,
            yticklabelsize=14,

            # Font styles
            titlefont=:bold,
            xticklabelfont=:regular,
            yticklabelfont=:regular,
            xlabelfont=:regular,
            ylabelfont=:regular,

            # Grid
            xgridwidth=1.25,
            ygridwidth=1.25,
            xgridcolor="white",
            ygridcolor="white",
            xminorgridcolor="white",
            yminorgridcolor="white",
            xminorgridvisible=false,
            xminorgridwidth=1.0,
            yminorgridvisible=false,
            yminorgridwidth=1.0,

            # Axis ticks
            minorticks=false,
            xticksvisible=false,
            yticksvisible=false,

            # Box
            rightspinevisible=false,
            leftspinevisible=false,
            topspinevisible=false,
            bottomspinevisible=false,
        ),
        Legend=(
            titlesize=15,
            labelsize=15,
            bgcolor="#E6E6EF",
        ),
        Lines=(
            linewidth=2,
        ),
        backgroundcolor="white",
        linewidth=1.25,
    )
    set_theme!(theme)
end

@doc raw"""
    colors()

Returns dictionary with personal color palette.
"""
function colors()
    col = Dict(
        :dark_black => "#000000",
        :black => "#000000",
        :light_black => "#05080F",
        :pale_black => "#1F1F1F",
        :dark_blue => "#2957A8",
        :blue => "#3876C0",
        :light_blue => "#81A9DA",
        :pale_blue => "#C0D4ED",
        :dark_green => "#2E5C0A",
        :green => "#468C12",
        :light_green => "#6EBC24",
        :pale_green => "#A9EB70",
        :dark_red => "#912E27",
        :red => "#CB4338",
        :light_red => "#D57A72",
        :pale_red => "#E8B5B0",
        :dark_gold => "#B68816",
        :gold => "#EBC21F",
        :light_gold => "#F2D769",
        :pale_gold => "#F7E6A1",
        :dark_purple => "#5E315E",
        :purple => "#934D93",
        :light_purple => "#BC7FBC",
        :pale_purple => "#D5AFD5"
    )

    # Initialize dictionary
    colors = Dict()

    # Loop through elements
    for (key, item) in col
        # Convert element to dictionary
        setindex!(colors, Colors.color(item), key)
    end # for

    return colors
end # function

