using StatsPlots  # StatsPlots re-exports Plots and adds boxplot, violin, etc.
import PlotlyJS
using PersistenceDiagrams
using PersistenceDiagrams: PersistenceImage, BettiCurve
using StatsBase: mean, std, countmap

# Color palette for drug categories
const DRUG_COLORS = Dict(
    "CONTROL" => :blue,
    "SPINOSAD" => :red,
    "ENDOSULFAN" => :green,
    "CIPERMETRINA" => :orange,
    "GLIFOSATO" => :purple
)

# Plot average persistence images per group
function plot_avg_persistence_images(pds, groups; size_pi::Int=10, sigma::Float64=0.1)
    unique_groups = unique(groups)
    n_groups = length(unique_groups)

    plots_list = []

    for g in unique_groups
        group_pds = pds[groups .== g]

        if !isempty(group_pds) && !all(isempty, group_pds)
            # Pass diagrams as array (not vcat) so PersistenceImage can iterate over intervals
            non_empty_group = filter(!isempty, group_pds)
            pi = PersistenceImage(non_empty_group; size=size_pi, sigma=sigma)
            images = [pi(pd) for pd in group_pds if !isempty(pd)]

            if !isempty(images)
                avg_img = mean(images)
                p = heatmap(avg_img,
                    title=string(g),
                    colorbar=false,
                    aspect_ratio=:equal)
                push!(plots_list, p)
            end
        end
    end

    if !isempty(plots_list)
        plot(plots_list..., layout=(1, n_groups), size=(200*n_groups, 200))
    else
        plot(title="No data available")
    end
end

# Plot Betti curves comparison by group
function plot_betti_curves_by_group(pds, groups; length_bc::Int=50)
    # Filter out empty diagrams for fitting the BettiCurve
    non_empty_pds = filter(!isempty, pds)
    bc = BettiCurve(non_empty_pds; length=length_bc)
    unique_groups = unique(groups)

    p = plot(xlabel="Filtration value", ylabel="Betti number",
             title="Betti Curves by Group", legend=:topright)

    for g in unique_groups
        group_pds = pds[groups .== g]
        curves = [bc(pd) for pd in group_pds if !isempty(pd)]

        if !isempty(curves)
            mean_curve = mean(curves)
            std_curve = std(curves)

            color = get(DRUG_COLORS, string(g), :gray)
            plot!(p, mean_curve,
                  ribbon=std_curve,
                  fillalpha=0.2,
                  label=string(g),
                  color=color,
                  linewidth=2)
        end
    end

    p
end

# Plot distance matrix heatmap sorted by group
function plot_distance_heatmap_sorted(D::Matrix, labels; title::String="Distance Matrix")
    sorted_idx = sortperm(labels)
    D_sorted = D[sorted_idx, sorted_idx]
    labels_sorted = labels[sorted_idx]

    n = length(labels)
    Plots.heatmap(D_sorted,
            xticks=(1:n, string.(labels_sorted)),
            yticks=(1:n, string.(labels_sorted)),
            xrotation=45,
            title=title,
            colorbar_title="Distance",
            size=(600, 500))
end

# 3D scatter plot with PlotlyJS
function plot_3d_scatter(X::Matrix, labels; title::String="3D Embedding")
    unique_labels = unique(labels)
    traces = PlotlyJS.GenericTrace[]

    for label in unique_labels
        idx = findall(==(label), labels)
        color = get(DRUG_COLORS, string(label), "gray")

        trace = PlotlyJS.scatter3d(
            x=X[1, idx],
            y=X[2, idx],
            z=X[3, idx],
            mode="markers",
            name=string(label),
            marker=PlotlyJS.attr(size=8, color=string(color))
        )
        push!(traces, trace)
    end

    layout = PlotlyJS.Layout(
        title=title,
        scene=PlotlyJS.attr(
            xaxis=PlotlyJS.attr(title="Dim 1"),
            yaxis=PlotlyJS.attr(title="Dim 2"),
            zaxis=PlotlyJS.attr(title="Dim 3")
        )
    )

    PlotlyJS.plot(traces, layout)
end

# Plot persistence diagram comparison
function plot_pd_comparison(pds, labels; max_per_group::Int=3)
    unique_labels = unique(labels)
    n_groups = length(unique_labels)

    plots_list = []

    for label in unique_labels
        group_idx = findall(==(label), labels)
        n_to_plot = min(max_per_group, length(group_idx))

        for i in 1:n_to_plot
            pd = pds[group_idx[i]]
            if !isempty(pd)
                p = plot(pd,
                    persistence=true,
                    title="$(label) #$(i)",
                    legend=false)
                push!(plots_list, p)
            end
        end
    end

    n_plots = length(plots_list)
    ncols = min(n_plots, n_groups)
    nrows = ceil(Int, n_plots / ncols)

    plot(plots_list..., layout=(nrows, ncols), size=(300*ncols, 250*nrows))
end

# Plot feature distributions by group (boxplot)
function plot_feature_boxplot(values, groups, feature_name::String)
    unique_groups = unique(groups)

    p = plot(xlabel="Group", ylabel=feature_name,
             title="$(feature_name) by Group")

    for (i, g) in enumerate(unique_groups)
        group_values = values[groups .== g]
        color = get(DRUG_COLORS, string(g), :gray)

        boxplot!(p, fill(string(g), length(group_values)), group_values,
                 label=string(g), color=color, alpha=0.7)
    end

    p
end

# Plot rich stats comparison
function plot_stats_comparison(stats_list, labels)
    # Extract feature names from first non-empty stats
    feature_names = collect(keys(stats_list[1]))

    n_features = length(feature_names)
    plots_list = []

    for fname in feature_names
        values = [s[fname] for s in stats_list]
        p = plot_feature_boxplot(values, labels, string(fname))
        push!(plots_list, p)
    end

    ncols = 3
    nrows = ceil(Int, n_features / ncols)
    plot(plots_list..., layout=(nrows, ncols), size=(400*ncols, 300*nrows))
end

# Plot MDS embedding (2D version)
function plot_mds_2d(X::Matrix, labels; title::String="MDS Embedding")
    unique_labels = unique(labels)

    p = Plots.plot(xlabel="Dim 1", ylabel="Dim 2", title=title,
             legend=:topright, aspect_ratio=:equal)

    for label in unique_labels
        idx = findall(==(label), labels)
        color = get(DRUG_COLORS, string(label), :gray)

        Plots.scatter!(p, X[1, idx], X[2, idx],
                 label=string(label),
                 color=color,
                 markersize=8,
                 markerstrokewidth=1)
    end

    p
end

# Confusion matrix heatmap
"""
    plot_confusion_matrix(y_true, y_pred; normalize=false, title="Confusion Matrix")

Heatmap visualization of confusion matrix with annotations.
"""
function plot_confusion_matrix(y_true, y_pred;
                                normalize::Bool=false,
                                title::String="Confusion Matrix")
    # Build confusion matrix
    labels = sort(unique(vcat(y_true, y_pred)))
    n = length(labels)
    label_to_idx = Dict(l => i for (i, l) in enumerate(labels))

    cm = zeros(n, n)
    for (t, p) in zip(y_true, y_pred)
        cm[label_to_idx[t], label_to_idx[p]] += 1
    end

    if normalize
        row_sums = sum(cm, dims=2)
        cm = cm ./ max.(row_sums, 1)  # Avoid division by zero
        colorbar_title = "Proportion"
    else
        colorbar_title = "Count"
    end

    # Create heatmap
    plt = Plots.heatmap(
        cm,
        xticks=(1:n, string.(labels)),
        yticks=(1:n, string.(labels)),
        xlabel="Predicted",
        ylabel="True",
        title=title,
        color=:Blues,
        colorbar_title=colorbar_title,
        aspect_ratio=:equal,
        xrotation=45,
        size=(500, 450)
    )

    # Add text annotations
    for i in 1:n, j in 1:n
        val = normalize ? round(cm[i,j], digits=2) : Int(cm[i,j])
        text_color = cm[i,j] > maximum(cm)/2 ? :white : :black
        Plots.annotate!(plt, j, i, Plots.text(string(val), 9, text_color))
    end

    plt
end

# Feature importance bar plot
"""
    plot_feature_importance(importances, feature_names; top_n=10, title="Feature Importance")

Horizontal bar plot of feature importances, showing top_n most important features.
"""
function plot_feature_importance(importances::Vector, feature_names;
                                  top_n::Int=10,
                                  title::String="Feature Importance")
    sorted_idx = sortperm(importances, rev=true)
    top_idx = sorted_idx[1:min(top_n, length(sorted_idx))]

    # Reverse for horizontal bar plot (top feature at top)
    top_idx_rev = reverse(top_idx)

    Plots.bar(
        importances[top_idx_rev],
        yticks=(1:length(top_idx_rev), string.(feature_names[top_idx_rev])),
        orientation=:h,
        xlabel="Importance (Accuracy Drop)",
        title=title,
        legend=false,
        color=:steelblue,
        size=(500, 300 + 20 * length(top_idx_rev))
    )
end
