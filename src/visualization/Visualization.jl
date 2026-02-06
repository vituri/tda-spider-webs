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
    plot_confusion_matrix,
    plot_feature_importance,
    # Separability analysis plots
    plot_alpha_optimization,
    plot_roc_curve,
    plot_separability_comparison,
    plot_silhouette_by_class,
    plot_binary_classification_summary,
    plot_pairwise_group_heatmap

end
