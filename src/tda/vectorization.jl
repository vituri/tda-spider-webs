using PersistenceDiagrams
using PersistenceDiagrams: PersistenceImage, BettiCurve, Landscape, Silhouette, Wasserstein, Bottleneck
using Ripserer: persistence, birth, death
using StatsBase: mean, std, quantile

# Persistence entropy calculation
function persistence_entropy(persistences::Vector{<:Real})
    if isempty(persistences) || all(p -> p <= 0, persistences)
        return 0.0
    end

    ps = filter(>(0), persistences)
    R = sum(ps)
    if R == 0
        return 0.0
    end

    probs = ps ./ R
    -sum(p * log(p) for p in probs if p > 0)
end

# Rich statistics from persistence diagram
function rich_stats(pd)
    # Default empty result
    empty_result = (
        n_features = 0,
        total_persistence = 0.0,
        mean_persistence = 0.0,
        std_persistence = 0.0,
        max_persistence = 0.0,
        q25 = 0.0,
        q50 = 0.0,
        q75 = 0.0,
        q90 = 0.0,
        entropy = 0.0,
        mean_birth = 0.0,
        birth_range = 0.0,
    )

    # Handle empty diagrams
    if isempty(pd)
        return empty_result
    end

    # Safely extract persistence and birth values
    try
        pers = persistence.(pd)
        births = birth.(pd)

        # Filter to valid finite values
        valid_pers = filter(isfinite, pers)
        valid_births = filter(isfinite, births)

        # If no valid values, return empty result
        if isempty(valid_pers) || isempty(valid_births)
            return empty_result
        end

        (
            n_features = length(pd),
            total_persistence = sum(valid_pers),
            mean_persistence = mean(valid_pers),
            std_persistence = length(valid_pers) > 1 ? std(valid_pers) : 0.0,
            max_persistence = maximum(valid_pers),
            q25 = quantile(valid_pers, 0.25),
            q50 = quantile(valid_pers, 0.50),
            q75 = quantile(valid_pers, 0.75),
            q90 = quantile(valid_pers, 0.90),
            entropy = persistence_entropy(valid_pers),
            mean_birth = mean(valid_births),
            birth_range = maximum(valid_births) - minimum(valid_births),
        )
    catch
        return empty_result
    end
end

# Vectorize a persistence diagram using multiple representations
function vectorize_diagram(pd;
    pi_size::Int=10,
    pi_sigma::Float64=0.1,
    betti_length::Int=50,
    n_landscapes::Int=3,
    silhouette_length::Int=50
)
    features = Float64[]

    if isempty(pd)
        # Return zero vector with expected size
        n_pi = pi_size * pi_size
        n_betti = betti_length
        n_landscapes_total = n_landscapes * 50  # Default landscape length
        n_silhouette = silhouette_length
        n_stats = 12  # Number of rich_stats fields

        return zeros(n_pi + n_betti + n_landscapes_total + n_silhouette + n_stats)
    end

    # Persistence Image
    try
        pi = PersistenceImage(pd; size=pi_size, sigma=pi_sigma)
        append!(features, vec(pi(pd)))
    catch
        append!(features, zeros(pi_size * pi_size))
    end

    # Betti Curve
    try
        bc = BettiCurve(; length=betti_length)
        append!(features, bc(pd))
    catch
        append!(features, zeros(betti_length))
    end

    # Persistence Landscapes
    for k in 1:n_landscapes
        try
            lk = Landscape(k)
            append!(features, lk(pd))
        catch
            append!(features, zeros(50))  # Default landscape length
        end
    end

    # Silhouette
    try
        sil = Silhouette(; length=silhouette_length)
        append!(features, sil(pd))
    catch
        append!(features, zeros(silhouette_length))
    end

    # Rich statistics
    stats = rich_stats(pd)
    append!(features, collect(values(stats)))

    features
end

# Distance matrix between persistence diagrams
function pd_distance_matrix(pds::Vector; metric=Wasserstein(q=2))
    n = length(pds)
    D = zeros(n, n)

    for i in 1:n
        for j in (i+1):n
            d = metric(pds[i], pds[j])
            D[i, j] = d
            D[j, i] = d
        end
    end

    D
end

# KNN classification using Wasserstein distance
function knn_wasserstein(train_pds, train_labels, test_pd; k::Int=3)
    dists = [Wasserstein(1, 2)(test_pd, pd) for pd in train_pds]
    sorted_idx = sortperm(dists)
    top_k_labels = train_labels[sorted_idx[1:k]]

    # Majority vote
    label_counts = Dict{eltype(train_labels), Int}()
    for label in top_k_labels
        label_counts[label] = get(label_counts, label, 0) + 1
    end

    findmax(label_counts)[2]
end

# Leave-one-out cross-validation with KNN Wasserstein
function loocv_knn_wasserstein(pds, labels; k::Int=3)
    n = length(pds)
    predictions = similar(labels)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        train_pds = pds[train_idx]
        train_labels = labels[train_idx]

        predictions[i] = knn_wasserstein(train_pds, train_labels, pds[i]; k=k)
    end

    accuracy = sum(predictions .== labels) / n
    (predictions=predictions, accuracy=accuracy)
end
