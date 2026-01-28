using MultivariateStats
using HypothesisTests
using StatsBase: mode, countmap, mean, std
using DataFrames

# MDS embedding from distance matrix
function mds_embedding(D::Matrix; maxoutdim::Int=3)
    mds_model = fit(MDS, D; maxoutdim=maxoutdim, distances=true)
    X_embedded = predict(mds_model)
    (model=mds_model, embedding=X_embedded)
end

# Confusion matrix
function confusion_matrix(y_true, y_pred)
    labels = sort(unique(vcat(y_true, y_pred)))
    n = length(labels)
    label_to_idx = Dict(l => i for (i, l) in enumerate(labels))

    cm = zeros(Int, n, n)
    for (t, p) in zip(y_true, y_pred)
        cm[label_to_idx[t], label_to_idx[p]] += 1
    end

    (matrix=cm, labels=labels)
end

# Classification report
function classification_report(y_true, y_pred)
    cm_result = confusion_matrix(y_true, y_pred)
    cm = cm_result.matrix
    labels = cm_result.labels
    n = length(labels)

    report = Dict{String, Any}()

    for (i, label) in enumerate(labels)
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = sum(cm) - tp - fp - fn

        precision = tp + fp > 0 ? tp / (tp + fp) : 0.0
        recall = tp + fn > 0 ? tp / (tp + fn) : 0.0
        f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0

        report[string(label)] = (
            precision=precision,
            recall=recall,
            f1=f1,
            support=tp + fn
        )
    end

    overall_accuracy = sum(diag(cm)) / sum(cm)
    report["accuracy"] = overall_accuracy
    report["confusion_matrix"] = cm
    report["labels"] = labels

    report
end

# Diagonal extraction helper
function diag(M::Matrix)
    [M[i, i] for i in 1:min(size(M)...)]
end

# Stratified k-fold cross-validation indices
function stratified_kfold(labels; k::Int=5, shuffle::Bool=true, seed::Int=42)
    if shuffle
        Random.seed!(seed)
    end

    n = length(labels)
    unique_labels = unique(labels)
    folds = [Int[] for _ in 1:k]

    for label in unique_labels
        indices = findall(==(label), labels)
        if shuffle
            indices = indices[randperm(length(indices))]
        end

        for (i, idx) in enumerate(indices)
            fold_idx = mod1(i, k)
            push!(folds[fold_idx], idx)
        end
    end

    # Generate train/test splits
    splits = []
    for i in 1:k
        test_idx = folds[i]
        train_idx = vcat([folds[j] for j in 1:k if j != i]...)
        push!(splits, (train=train_idx, test=test_idx))
    end

    splits
end

# Cross-validation evaluation
function cross_validate(predict_fn, X, y; k::Int=5, shuffle::Bool=true, seed::Int=42)
    splits = stratified_kfold(y; k=k, shuffle=shuffle, seed=seed)
    accuracies = Float64[]

    for (train_idx, test_idx) in splits
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]

        y_pred = predict_fn(X_train, y_train, X_test)
        acc = sum(y_pred .== y_test) / length(y_test)
        push!(accuracies, acc)
    end

    (mean=mean(accuracies), std=std(accuracies), scores=accuracies)
end

# Simple KNN classifier
function knn_predict(X_train, y_train, X_test; k::Int=3)
    predictions = similar(y_train, size(X_test, 1))

    for (i, x_test) in enumerate(eachrow(X_test))
        # Compute distances to all training points
        dists = [sum((x_test .- x_train).^2) for x_train in eachrow(X_train)]
        sorted_idx = sortperm(dists)
        top_k_labels = y_train[sorted_idx[1:k]]

        # Majority vote
        label_counts = countmap(top_k_labels)
        predictions[i] = argmax(label_counts)
    end

    predictions
end

# Kruskal-Wallis test for group differences
function test_group_differences(feature_values, groups)
    unique_groups = unique(groups)
    group_data = [feature_values[groups .== g] for g in unique_groups]

    # Kruskal-Wallis test
    kw_test = KruskalWallisTest(group_data...)

    (test=kw_test, p_value=pvalue(kw_test), groups=unique_groups)
end

# Permutation test for pairwise comparison
function permutation_test(group1, group2; n_permutations::Int=10000, seed::Int=42)
    Random.seed!(seed)

    observed_diff = abs(mean(group1) - mean(group2))
    combined = vcat(group1, group2)
    n1 = length(group1)

    count = 0
    for _ in 1:n_permutations
        shuffled = combined[randperm(length(combined))]
        perm_diff = abs(mean(shuffled[1:n1]) - mean(shuffled[n1+1:end]))
        if perm_diff >= observed_diff
            count += 1
        end
    end

    p_value = count / n_permutations
    (p_value=p_value, observed_diff=observed_diff, n_permutations=n_permutations)
end

# Feature importance via permutation
function feature_importance_permutation(predict_fn, X, y; n_repeats::Int=10, seed::Int=42)
    Random.seed!(seed)

    n_features = size(X, 2)
    baseline_acc = sum(predict_fn(X, y, X) .== y) / length(y)

    importances = zeros(n_features)

    for f in 1:n_features
        drop_accs = Float64[]
        for _ in 1:n_repeats
            X_permuted = copy(X)
            X_permuted[:, f] = X_permuted[randperm(size(X, 1)), f]
            perm_acc = sum(predict_fn(X, y, X_permuted) .== y) / length(y)
            push!(drop_accs, baseline_acc - perm_acc)
        end
        importances[f] = mean(drop_accs)
    end

    importances
end

# Convert rich_stats NamedTuples to feature matrix
"""
    stats_to_matrix(stats_list) -> (Matrix, Vector{Symbol})

Convert a vector of NamedTuples (from rich_stats) to a feature matrix.
Returns (X, feature_names) where X is n x p matrix.
"""
function stats_to_matrix(stats_list)
    n = length(stats_list)
    feature_names = collect(keys(stats_list[1]))
    p = length(feature_names)

    X = zeros(n, p)
    for (i, s) in enumerate(stats_list)
        X[i, :] = collect(values(s))
    end

    X, feature_names
end

# Cohen's d effect size
"""
    cohens_d(group1, group2) -> Float64

Calculate Cohen's d effect size between two groups.
Interpretation: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large
"""
function cohens_d(group1::Vector, group2::Vector)
    n1, n2 = length(group1), length(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Pooled standard deviation
    s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2))

    s_pooled == 0 ? 0.0 : (m1 - m2) / s_pooled
end

# Effect size interpretation
function effect_size_interpretation(d::Real)
    abs_d = abs(d)
    if abs_d < 0.2
        "negligible"
    elseif abs_d < 0.5
        "small"
    elseif abs_d < 0.8
        "medium"
    else
        "large"
    end
end

# Pairwise drug comparison with effect sizes
"""
    pairwise_drug_comparison(feature_values, groups; control_group="CONTROL", feature_name="feature")

Compare each drug group to control with effect sizes and p-values.
Returns DataFrame with comparisons.
"""
function pairwise_drug_comparison(feature_values, groups;
                                   control_group="CONTROL",
                                   feature_name="feature")
    unique_groups = unique(groups)
    control_idx = findall(==(control_group), groups)
    control_vals = feature_values[control_idx]

    results = NamedTuple[]

    for g in unique_groups
        g == control_group && continue

        drug_idx = findall(==(g), groups)
        drug_vals = feature_values[drug_idx]

        # Effect size
        d = cohens_d(drug_vals, control_vals)

        # Permutation test p-value
        perm_result = permutation_test(drug_vals, control_vals)

        push!(results, (
            drug = g,
            feature = feature_name,
            mean_drug = mean(drug_vals),
            mean_control = mean(control_vals),
            diff_percent = (mean(drug_vals) - mean(control_vals)) / mean(control_vals) * 100,
            cohens_d = d,
            effect_size = effect_size_interpretation(d),
            p_value = perm_result.p_value
        ))
    end

    DataFrame(results)
end
