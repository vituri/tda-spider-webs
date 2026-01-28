module TDA

include("functions.jl")
include("vectorization.jl")

export persistence_diagram,
    plot_barcode,
    plot_pd,
    modify_array,
    dist_to_line,
    dist_to_point,
    cubical_pd,
    rips_pd,
    # Vectorization exports
    persistence_entropy,
    rich_stats,
    vectorize_diagram,
    pd_distance_matrix,
    knn_wasserstein,
    loocv_knn_wasserstein

end