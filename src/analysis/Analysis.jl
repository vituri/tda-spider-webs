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
    stats_to_matrix,
    cohens_d,
    effect_size_interpretation,
    pairwise_drug_comparison,
    # Distance combination
    combine_distances,
    loocv_knn_distance,
    optimize_alpha,
    # Binary classification
    binary_classification_control_vs_rest,
    roc_curve_control,
    # Separability metrics
    within_between_ratio,
    silhouette_by_class,
    pairwise_group_distances,
    # PERMANOVA tests
    permanova,
    permanova_control_vs_drugs,
    drug_equivalence_test,
    pairwise_drug_permutation_tests,
    pairwise_confusion_analysis

end