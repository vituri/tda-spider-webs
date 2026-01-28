module Visualization

include("plots.jl")

export DRUG_COLORS,
    plot_avg_persistence_images,
    plot_betti_curves_by_group,
    plot_distance_heatmap_sorted,
    plot_3d_scatter,
    plot_pd_comparison,
    plot_feature_boxplot,
    plot_stats_comparison,
    plot_mds_2d,
    # New exports
    plot_confusion_matrix,
    plot_feature_importance

end
