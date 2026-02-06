using MultivariateStats
using HypothesisTests
using StatsBase: mode, countmap, mean, std, quantile
using DataFrames
using Random
using LinearAlgebra

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

# ============================================================================
# DISTANCE COMBINATION
# ============================================================================

"""
    combine_distances(D1::Matrix, D2::Matrix; alpha::Float64=0.5) -> Matrix

Combine two distance matrices with weighted average.
D1 and D2 are first normalized to [0,1] range, then combined:
    D_combined = alpha * D1_norm + (1-alpha) * D2_norm

Parameters:
- alpha: weight for first distance matrix (0 = only D2, 1 = only D1)
"""
function combine_distances(D1::Matrix, D2::Matrix; alpha::Float64=0.5)
    # Normalize to [0,1] range
    D1_range = maximum(D1) - minimum(D1)
    D2_range = maximum(D2) - minimum(D2)

    D1_norm = D1_range > 0 ? (D1 .- minimum(D1)) ./ D1_range : D1
    D2_norm = D2_range > 0 ? (D2 .- minimum(D2)) ./ D2_range : D2

    alpha * D1_norm + (1 - alpha) * D2_norm
end

"""
    loocv_knn_distance(D::Matrix, labels; k::Int=3) -> Float64

Leave-one-out cross-validation using precomputed distance matrix.
"""
function loocv_knn_distance(D::Matrix, labels; k::Int=3)
    n = length(labels)
    correct = 0

    for i in 1:n
        # Get distances from sample i to all others
        dists = copy(D[i, :])
        dists[i] = Inf  # Exclude self

        # Find k nearest neighbors
        sorted_idx = sortperm(dists)
        top_k_labels = labels[sorted_idx[1:k]]

        # Majority vote
        label_counts = countmap(top_k_labels)
        prediction = argmax(label_counts)

        if prediction == labels[i]
            correct += 1
        end
    end

    correct / n
end

"""
    optimize_alpha(D1::Matrix, D2::Matrix, labels; alpha_range=0.0:0.1:1.0, k::Int=3)

Find optimal alpha by grid search using LOOCV accuracy.
"""
function optimize_alpha(D1::Matrix, D2::Matrix, labels;
                        alpha_range=0.0:0.1:1.0, k::Int=3)
    results = NamedTuple[]

    for alpha in alpha_range
        D_combined = combine_distances(D1, D2; alpha=alpha)
        acc = loocv_knn_distance(D_combined, labels; k=k)
        push!(results, (alpha=alpha, accuracy=acc))
    end

    best = results[argmax([r.accuracy for r in results])]
    (best=best, all_results=results)
end

# ============================================================================
# BINARY CLASSIFICATION (CONTROL VS REST)
# ============================================================================

"""
    binary_classification_control_vs_rest(X, labels; n_bootstrap=1000, k=3, seed=42)

Binary classification: CONTROL vs all other drugs combined.
Returns accuracy with 95% confidence intervals via bootstrap.
"""
function binary_classification_control_vs_rest(X, labels;
                                                n_bootstrap::Int=1000,
                                                k::Int=3,
                                                seed::Int=42)
    Random.seed!(seed)

    # Create binary labels
    binary_labels = [l == "CONTROL" ? "CONTROL" : "DRUG" for l in labels]

    # LOOCV for point estimate
    n = size(X, 1)
    predictions = similar(binary_labels)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        pred = knn_predict(X[train_idx, :], binary_labels[train_idx], X[i:i, :]; k=k)
        predictions[i] = pred[1]
    end

    accuracy = sum(predictions .== binary_labels) / n

    # Bootstrap confidence intervals
    bootstrap_accs = Float64[]
    for _ in 1:n_bootstrap
        boot_idx = rand(1:n, n)
        boot_acc = sum(predictions[boot_idx] .== binary_labels[boot_idx]) / n
        push!(bootstrap_accs, boot_acc)
    end

    ci_lower = quantile(bootstrap_accs, 0.025)
    ci_upper = quantile(bootstrap_accs, 0.975)

    # Compute sensitivity and specificity
    control_idx = findall(==("CONTROL"), binary_labels)
    drug_idx = findall(==("DRUG"), binary_labels)

    sensitivity = sum(predictions[control_idx] .== "CONTROL") / length(control_idx)
    specificity = sum(predictions[drug_idx] .== "DRUG") / length(drug_idx)

    (accuracy=accuracy, ci_lower=ci_lower, ci_upper=ci_upper,
     sensitivity=sensitivity, specificity=specificity,
     predictions=predictions, binary_labels=binary_labels)
end

"""
    roc_curve_control(X, labels; n_thresholds=100)

Generate ROC curve for detecting CONTROL samples.
Uses distance to Control centroid as the decision function.
Returns (fpr, tpr, auc, thresholds).
"""
function roc_curve_control(X, labels; n_thresholds::Int=100)
    # Compute centroid of CONTROL group
    control_idx = findall(==("CONTROL"), labels)
    control_centroid = vec(mean(X[control_idx, :], dims=1))

    # Distance from each sample to control centroid
    distances = [sqrt(sum((X[i, :] .- control_centroid).^2)) for i in 1:size(X, 1)]

    # Binary labels (1 = CONTROL, 0 = DRUG)
    binary = [l == "CONTROL" ? 1 : 0 for l in labels]

    # Generate ROC curve points
    thresholds = range(minimum(distances) - 0.1, maximum(distances) + 0.1, length=n_thresholds)
    tpr_list = Float64[]  # True Positive Rate (Sensitivity)
    fpr_list = Float64[]  # False Positive Rate (1 - Specificity)

    for thresh in thresholds
        # Predict CONTROL if distance < threshold
        preds = [d < thresh ? 1 : 0 for d in distances]

        tp = sum((preds .== 1) .& (binary .== 1))
        fp = sum((preds .== 1) .& (binary .== 0))
        fn = sum((preds .== 0) .& (binary .== 1))
        tn = sum((preds .== 0) .& (binary .== 0))

        tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0.0

        push!(tpr_list, tpr)
        push!(fpr_list, fpr)
    end

    # Compute AUC using trapezoidal rule
    sorted_idx = sortperm(fpr_list)
    fpr_sorted = fpr_list[sorted_idx]
    tpr_sorted = tpr_list[sorted_idx]

    auc = 0.0
    for i in 2:length(fpr_sorted)
        auc += (fpr_sorted[i] - fpr_sorted[i-1]) * (tpr_sorted[i] + tpr_sorted[i-1]) / 2
    end

    (fpr=fpr_list, tpr=tpr_list, auc=auc, thresholds=collect(thresholds))
end

# ============================================================================
# SEPARABILITY METRICS
# ============================================================================

"""
    within_between_ratio(D::Matrix, labels)

Compute ratio of within-class to between-class distances.
Lower ratio indicates better class separation; ratio > 0.8 suggests overlapping classes.
"""
function within_between_ratio(D::Matrix, labels)
    unique_labels = unique(labels)

    within_dists = Float64[]
    between_dists = Float64[]

    for i in 1:size(D, 1)
        for j in (i+1):size(D, 1)
            if labels[i] == labels[j]
                push!(within_dists, D[i, j])
            else
                push!(between_dists, D[i, j])
            end
        end
    end

    mean_within = isempty(within_dists) ? 0.0 : mean(within_dists)
    mean_between = isempty(between_dists) ? 0.0 : mean(between_dists)
    ratio = mean_between > 0 ? mean_within / mean_between : 0.0

    interpretation = ratio < 0.5 ? "well separated" :
                     ratio < 0.8 ? "moderately separated" : "overlapping"

    (mean_within=mean_within, mean_between=mean_between, ratio=ratio,
     interpretation=interpretation,
     within_dists=within_dists, between_dists=between_dists)
end

"""
    silhouette_by_class(D::Matrix, labels)

Compute silhouette scores for each sample and summarize by class.
Score > 0.5 = good separation; < 0.25 = poor (overlapping).
"""
function silhouette_by_class(D::Matrix, labels)
    unique_labels = unique(labels)
    n = length(labels)

    silhouette_scores = Float64[]

    for i in 1:n
        own_label = labels[i]
        own_idx = findall(==(own_label), labels)
        other_labels = filter(!=(own_label), unique_labels)

        # a(i) = mean distance to own cluster (excluding self)
        own_dists = [D[i, j] for j in own_idx if j != i]
        a_i = isempty(own_dists) ? 0.0 : mean(own_dists)

        # b(i) = min mean distance to nearest other cluster
        b_i = Inf
        for other_label in other_labels
            other_idx = findall(==(other_label), labels)
            if !isempty(other_idx)
                other_dists = [D[i, j] for j in other_idx]
                mean_dist = mean(other_dists)
                b_i = min(b_i, mean_dist)
            end
        end

        # Silhouette score
        s_i = isinf(b_i) ? 0.0 : (b_i - a_i) / max(a_i, b_i)
        push!(silhouette_scores, isnan(s_i) ? 0.0 : s_i)
    end

    # Summarize by class
    by_class = Dict{String, NamedTuple}()
    for label in unique_labels
        idx = findall(==(label), labels)
        scores = silhouette_scores[idx]
        by_class[string(label)] = (
            mean = mean(scores),
            std = length(scores) > 1 ? std(scores) : 0.0,
            min = minimum(scores),
            max = maximum(scores)
        )
    end

    (scores=silhouette_scores, by_class=by_class, overall_mean=mean(silhouette_scores))
end

"""
    pairwise_group_distances(D::Matrix, labels)

Compute mean distance between all pairs of groups.
"""
function pairwise_group_distances(D::Matrix, labels)
    unique_labels = sort(unique(labels))
    n_groups = length(unique_labels)

    result = DataFrame(
        group1 = String[],
        group2 = String[],
        mean_distance = Float64[],
        std_distance = Float64[],
        n_pairs = Int[]
    )

    for i in 1:n_groups
        for j in i:n_groups
            g1, g2 = unique_labels[i], unique_labels[j]
            idx1 = findall(==(g1), labels)
            idx2 = findall(==(g2), labels)

            dists = Float64[]
            for a in idx1
                for b in idx2
                    if a != b
                        push!(dists, D[a, b])
                    end
                end
            end

            if !isempty(dists)
                push!(result, (
                    group1 = string(g1),
                    group2 = string(g2),
                    mean_distance = mean(dists),
                    std_distance = length(dists) > 1 ? std(dists) : 0.0,
                    n_pairs = length(dists)
                ))
            end
        end
    end

    result
end

# ============================================================================
# PERMANOVA TESTS
# ============================================================================

"""
    permanova(D::Matrix, labels; n_permutations=9999, seed=42)

PERMANOVA test: Tests if group centroids differ in multivariate space.
Works directly on distance matrix (e.g., Wasserstein distances).
"""
function permanova(D::Matrix, labels; n_permutations::Int=9999, seed::Int=42)
    Random.seed!(seed)

    function compute_pseudo_f(D, groups)
        n = size(D, 1)
        unique_groups = unique(groups)
        k = length(unique_groups)

        # Total sum of squares (based on distances)
        ss_total = sum(D.^2) / (2 * n)

        # Within-group sum of squares
        ss_within = 0.0
        for g in unique_groups
            idx = findall(==(g), groups)
            ng = length(idx)
            if ng > 1
                D_sub = D[idx, idx]
                ss_within += sum(D_sub.^2) / (2 * ng)
            end
        end

        # Between-group sum of squares
        ss_between = ss_total - ss_within

        # Degrees of freedom
        df_between = k - 1
        df_within = n - k

        # Pseudo-F (handle edge cases)
        (df_between <= 0 || df_within <= 0 || ss_within <= 0) && return 0.0

        f_stat = (ss_between / df_between) / (ss_within / df_within)
        f_stat
    end

    observed_f = compute_pseudo_f(D, labels)

    # Permutation test
    count_greater = 0
    for _ in 1:n_permutations
        perm_groups = labels[randperm(length(labels))]
        perm_f = compute_pseudo_f(D, perm_groups)
        if perm_f >= observed_f
            count_greater += 1
        end
    end

    p_value = (count_greater + 1) / (n_permutations + 1)

    (f_statistic=observed_f, p_value=p_value, n_permutations=n_permutations)
end

"""
    permanova_control_vs_drugs(D::Matrix, labels; n_permutations=9999, seed=42)

PERMANOVA test: Is the multivariate centroid of CONTROL different from DRUGS?
"""
function permanova_control_vs_drugs(D::Matrix, labels; n_permutations::Int=9999, seed::Int=42)
    # Binary grouping
    binary_labels = [l == "CONTROL" ? "CONTROL" : "DRUG" for l in labels]
    permanova(D, binary_labels; n_permutations=n_permutations, seed=seed)
end

"""
    drug_equivalence_test(D::Matrix, labels; n_permutations=9999, seed=42)

PERMANOVA among drug groups only (excluding CONTROL).
Tests: Are the drug multivariate centroids different from each other?
High p-value (>0.05) suggests drugs are NOT distinguishable.
"""
function drug_equivalence_test(D::Matrix, labels; n_permutations::Int=9999, seed::Int=42)
    # Filter to drugs only
    drug_idx = findall(!=("CONTROL"), labels)
    D_drugs = D[drug_idx, drug_idx]
    labels_drugs = labels[drug_idx]

    result = permanova(D_drugs, labels_drugs; n_permutations=n_permutations, seed=seed)

    interpretation = result.p_value >= 0.05 ?
        "Drugs NOT significantly different (supports non-separability)" :
        "Some drug differences detected"

    (f_statistic=result.f_statistic, p_value=result.p_value,
     n_permutations=result.n_permutations, interpretation=interpretation)
end

"""
    pairwise_drug_permutation_tests(feature_values, labels; n_permutations=10000, seed=42)

Test each drug pair: Can we reject that they come from the same distribution?
High p-values indicate drugs are NOT distinguishable.
"""
function pairwise_drug_permutation_tests(feature_values, labels;
                                          n_permutations::Int=10000,
                                          seed::Int=42)
    Random.seed!(seed)

    drug_labels = filter(!=("CONTROL"), unique(labels))
    n_drugs = length(drug_labels)

    results = NamedTuple[]

    for i in 1:n_drugs
        for j in (i+1):n_drugs
            d1, d2 = drug_labels[i], drug_labels[j]

            vals1 = feature_values[labels .== d1]
            vals2 = feature_values[labels .== d2]

            # Observed difference in means
            observed_diff = abs(mean(vals1) - mean(vals2))

            # Permutation test
            combined = vcat(vals1, vals2)
            n1 = length(vals1)
            count_extreme = 0

            for _ in 1:n_permutations
                shuffled = combined[randperm(length(combined))]
                perm_diff = abs(mean(shuffled[1:n1]) - mean(shuffled[n1+1:end]))
                if perm_diff >= observed_diff
                    count_extreme += 1
                end
            end

            p_value = (count_extreme + 1) / (n_permutations + 1)

            push!(results, (
                drug1 = string(d1),
                drug2 = string(d2),
                mean_diff = observed_diff,
                p_value = p_value,
                significant = p_value < 0.05,
                interpretation = p_value >= 0.05 ? "NOT distinguishable" : "distinguishable"
            ))
        end
    end

    DataFrame(results)
end

"""
    pairwise_confusion_analysis(y_true, y_pred)

Analyze which classes are most confused with each other.
Returns confusion rates between all pairs.
"""
function pairwise_confusion_analysis(y_true, y_pred)
    labels = sort(unique(vcat(y_true, y_pred)))
    n_labels = length(labels)

    # Build confusion matrix
    cm = zeros(n_labels, n_labels)
    label_to_idx = Dict(l => i for (i, l) in enumerate(labels))

    for (t, p) in zip(y_true, y_pred)
        cm[label_to_idx[t], label_to_idx[p]] += 1
    end

    # Compute confusion rates (off-diagonal / row total)
    confusion_pairs = NamedTuple[]

    for i in 1:n_labels
        for j in 1:n_labels
            if i != j
                row_total = sum(cm[i, :])
                confusion_rate = row_total > 0 ? cm[i, j] / row_total : 0.0
                push!(confusion_pairs, (
                    true_class = string(labels[i]),
                    predicted_class = string(labels[j]),
                    confusion_rate = confusion_rate * 100,
                    count = Int(cm[i, j])
                ))
            end
        end
    end

    # Sort by confusion rate (highest first)
    sort!(confusion_pairs, by = x -> x.confusion_rate, rev=true)

    DataFrame(confusion_pairs)
end
