module Analysis

using ..TDAweb
using Random

include("functions.jl")
include("classification.jl")

export pairwise_distance,
    plot_wing_with_pd,
    plot_heatmap,
    plot_scatter,
    # Classification exports
    mds_embedding,
    confusion_matrix,
    classification_report,
    stratified_kfold,
    cross_validate,
    knn_predict,
    test_group_differences,
    permutation_test,
    feature_importance_permutation,
    # New exports
    stats_to_matrix,
    cohens_d,
    effect_size_interpretation,
    pairwise_drug_comparison

end