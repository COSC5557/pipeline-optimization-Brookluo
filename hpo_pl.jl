# make sure local environment is activated
using Pkg
Pkg.activate(".")

using CSV
using Plots
# using MLDataUtils

# using Distributed
using DataFrames

# Add workers
# nprocs() == 1 && addprocs(10, exeflags=["--project=$(Base.active_project())"])
# workers()

using AutoMLPipeline
using Random

using Hyperopt
using Plots
using StatsPlots

using MLUtils
using Imbalance

# Load the data
df_red = CSV.read("winequality-white.csv", DataFrame)
# names = CSV.read("winequality.names", DataFrame)

# oversample(df_red[!, 1:end-1], df_red[!, end])

# a = df_red[!, end]
# [element => count(==(element),a) for element in sort(unique(a)) ]
checkbalance(df_red[!, end])
# split the data into training and testing
# Random.seed!(42)
rng = MersenneTwister(1234)
Xover, yover = smote(df_red[!, 1:end-1], df_red[!, end]; k=5,
                ratios=Dict(9 => 0.4,
                            3 => 0.4,
                            4 => 0.4,
                            8 => 0.4,
                            7 => 0.4,
                            5 => 0.663,
                            6 => 1.0), rng=rng)
checkbalance(yover)
4train, test = splitobs(hcat(Matrix(Xover), yover)', at=0.7, shuffle=true)
train = DataFrame(train', :auto)
test = DataFrame(test', :auto)
X_train = train[:, 1:end-1]
Y_train = Int.(train[:, end]) |> Vector
X_test = test[:, 1:end-1]
Y_test = Int.(test[:, end]) |> Vector
# head(x)=first(x,5)
# head(df_red)

# Define the model
#### Learners
rf = hp -> SKLearner("RandomForestClassifier", n_estimators=hp[1], max_depth=hp[2], max_features=hp[3], min_samples_split=Int(hp[4]),
                        min_samples_leaf=Int(hp[5]), random_state=0)
gb = hp -> SKLearner("GradientBoostingClassifier", n_estimators=Int(hp[1]), learning_rate=hp[2], max_depth=Int(hp[3]), min_samples_split=Int(hp[4]),
                        min_samples_leaf=Int(hp[5]), random_state=0)
svc = hp -> SKLearner("SVC", C=hp[1], kernel=String(hp[2]), degree=hp[3], random_state=0)

#### Decomposition
pca = n_components -> SKPreprocessor("PCA", Dict(:n_components => n_components, :random_state => 0))
ica = n_components -> SKPreprocessor("FastICA", Dict(:n_components => n_components, :whiten => true))

dim_reds = [:pca, :ica]
dim_reds_dict = Dict(:pca => pca, :ica => ica)
comp_range = 2:20

#### Scaler 
rb = SKPreprocessor("RobustScaler")
pt = SKPreprocessor("PowerTransformer")
norm = SKPreprocessor("Normalizer")
mx = SKPreprocessor("MinMaxScaler")
std = SKPreprocessor("StandardScaler")

scalers = [:rb, :pt, :norm, :mx, :std]
scalers_dict = Dict(:rb => rb, :pt => pt, :norm => norm, :mx => mx, :std => std)

#### categorical preprocessing
ohe = OneHotEncoder()

#### Column selector
catf = CatFeatureSelector()
numf = NumFeatureSelector()
disc = CatNumDiscriminator()


function preprocessing(pl_hp, X, Y)
    # first transform the data
    # use OneHotEncoder for categorical data and RobustScaler for numerical data
    # but technically, there should be no categorical data in this dataset
    Random.seed!(42)
    pl = @pipeline pl_hp[:scaler] |> pl_hp[:dim_red]
    X_train_trans = AutoMLPipeline.fit_transform!(pl, X, Y)
    X_test_trans = AutoMLPipeline.transform(pl, X_test)
    return (X_train_trans, X_test_trans)
end

# Define the pipeline
function HPOLearner(learner, X, Y)
    # we will use accuracy as the metric
    # we will use 5 fold cross validation
    Random.seed!(42)
    mean, sd, _ = crossvalidate(learner, X, Y, "accuracy_score", nfolds=5, verbose=false)
    # weight the accuracy by inverse variance
    # negative for minimization
    # return -mean / sd^2
    return -mean
end


function writeToCSV(filename, ho, colnames)
    params = permutedims(hcat(ho.history...))
    vals = [t[1] for t in ho.results]
    CSV.write(filename, DataFrame(hcat(-vals, params), colnames))
end

function writeTestResult(dir, ho_params_name, sampler_name, all_ho, all_cv)
    df_cv = DataFrame(all_cv)
    df_ho = DataFrame(permutedims(hcat([ho.minimizer for ho in all_ho]...)), ho_params_name[2:end])
    CSV.write(dir * "all_cv.csv", df_cv)
    CSV.write(dir * "all_ho_params.csv", df_ho)
    for i in 1:length(all_ho)
        writeToCSV(dir * sampler_name[i] * ".csv", all_ho[i], ho_params_name)
    end
end

function analyzePerf(model, dir, ho_params_name, sampler_name, all_ho, X_test, Y_test)
    Random.seed!(42)
    cv_res = []
    # total_mean = []
    p = boxplot()
    perfplot = plot()
    for (i, ho) in enumerate(all_ho)
        # plotHyperopt(ho, dir * sampler_name[i])
        push!(cv_res, crossvalidate(model(ho.minimizer), X_test, Y_test, "accuracy_score", nfolds=5, verbose=true))
        scores = [t[1] for t in ho.results]
        scores = scores[.!isnan.(scores)]
        # push!(total_mean, scores[.!isnan(scores)])
        boxplot!(p, [sampler_name[i]], -scores, ylabel="Accuracy", legend=false)
        plot!(perfplot, -scores, label=sampler_name[i], ylabel="Accuracy", legend=true)
    end
    # println(cv_res)
    plot!(p, legend=false, ylabel="Accuracy score")
    savefig(p, dir * "boxplot.png")
    plot!(perfplot, legend=true, ylabel="Accuracy score", xlims=(0, 50), ylims=(0.4, 0.7))
    savefig(perfplot, dir * "perfplot.png")
    writeTestResult(dir, ho_params_name, sampler_name, all_ho, cv_res)
end

# For Random Forest
println("Random Forest")
# Define hypterparameter function
HPO_rf = (hp, X, Y) -> HPOLearner(rf(hp), X, Y)
# Hyperparameter bounds for n_estimators, max_depth, max_feature
n_est_range = 10:500
max_depth_range = 1:30
max_feature_range = 1:30
min_samples_split_range = 2:50
min_samples_leaf_range = 1:50

# use Hyperband for optimization
println("Hyperband")
rf_hohb = @time @hyperopt for i = 50,
    sampler = Hyperband(R=100, η=3, inner=RandomSampler(rng)),
    n_est = n_est_range,
    max_depth = max_depth_range,
    max_feature = max_feature_range,
    min_samples_split = min_samples_split_range,
    min_samples_leaf = min_samples_leaf_range,
    n_comp = comp_range,
    dim_red = dim_reds,
    scaler = scalers

    if state !== nothing
        n_est, max_depth, max_feature, min_samples_split, min_samples_leaf, n_comp, dim_red, scaler = state
    end
    X_train_trans, X_test_trans = preprocessing(Dict(:dim_red => dim_reds_dict[dim_red](n_comp), :scaler => scalers_dict[scaler]), X_train, Y_train)
    HPO_rf([n_est, max_depth, max_feature, min_samples_split, min_samples_leaf], X_train_trans, Y_train), [n_est, max_depth, max_feature, min_samples_split, min_samples_leaf, n_comp, dim_red, scaler]
end

rf_params = ["accuracy", "n_estimators", "max_depth", "max_feature", "min_samples_split", "min_samples_leaf",
            "n_comp", "dim_red", "scaler"]
writeToCSV("./rf_hyperband_rs.csv", rf_hohb, rf_params)

# analyzePerf(rf, "./rf/", ["accuracy", "n_estimators", "max_depth", "max_feature"],
#     ["random_search", "hyperband_rs", "hyperband_bo"],
#     [hors, hohb, hohbbo], X_test_trans, Y_test)

# For Gradient Boosting
println("Gradient Boosting")
# Define hypterparameter function
HPO_gb = (hp, X, Y) -> HPOLearner(gb(hp), X, Y)
n_est_range = 10:500
lr_range = LinRange(0.01, 0.5, 100)
max_depth_range = 1:30


# use Hyperband for optimization
println("Hyperband")
gb_hohb = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=RandomSampler(rng)),
    n_est = n_est_range,
    lr = lr_range,
    max_depth = max_depth_range,
    min_samples_split = min_samples_split_range,
    min_samples_leaf = min_samples_leaf_range,
    n_comp = comp_range,
    dim_red = dim_reds,
    scaler = scalers

    if state !== nothing
        n_est, lr, max_depth, min_samples_split, min_samples_leaf, n_comp, dim_red, scaler = state
    end
    X_train_trans, X_test_trans = preprocessing(Dict(:dim_red => dim_reds_dict[dim_red](n_comp), :scaler => scalers_dict[scaler]), X_train, Y_train)
    @show HPO_gb([n_est, lr, max_depth, min_samples_split, min_samples_leaf], X_train_trans, Y_train), [n_est, lr, max_depth, min_samples_split, min_samples_leaf, n_comp, dim_red, scaler]
end

gb_params = ["accuracy", "n_estimators", "learning_rate", "max_depth", "min_samples_split", "min_samples_leaf", "n_comp", "dim_red", "scaler"]
writeToCSV("./gb_hyperband_rs.csv", gb_hohb, gb_params)

# For SVC
println("SVM")
# Define hypterparameter function
HPO_svc = (hp, X, Y) -> HPOLearner(svc(hp), X, Y)
C_range = LinRange(0.1, 10, 100)
kernel_range = ["linear", "poly", "rbf", "sigmoid"]
degree_range = 1:10

# use Hyperband for optimization
println("Hyperband")
svc_hohb = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=RandomSampler(rng)),
    C = C_range,
    kernel = kernel_range,
    degree = degree_range,
    n_comp = comp_range,
    dim_red = dim_reds,
    scaler = scalers

    if state !== nothing
        C, kernel, degree, n_comp, dim_red, scaler = state
    end
    X_train_trans, X_test_trans = preprocessing(Dict(:dim_red => dim_reds_dict[dim_red](n_comp), :scaler => scalers_dict[scaler]), X_train, Y_train)
    @show HPO_svc([C, kernel, degree], X_train_trans, Y_train), [C, kernel, degree, n_comp, dim_red, scaler]
end

# Cannot use hyperband with BO with categorical variables given
# the limit of the package

svc_params = ["accuracy", "C", "kernel", "degree", "n_comp", "dim_red", "scaler"]
writeToCSV("./svc_hyperband_rs.csv", svc_hohb, svc_params)


# Compare performance for all models
df_rf = CSV.read("./rf_hyperband_rs.csv", DataFrame)
df_gb = CSV.read("./gb_hyperband_rs.csv", DataFrame)
df_svm = CSV.read("./svc_hyperband_rs.csv", DataFrame)
p = boxplot(["Random Forest"], df_rf.accuracy, ylabel="Accuracy", legend=false)
boxplot!(p, ["Gradient Boosting"], df_gb.accuracy, ylabel="Accuracy", legend=false)
boxplot!(p, ["SVM"], df_svm.accuracy, ylabel="Accuracy", legend=false)
savefig(p, "all_hbrs_perf_boxplot.png")

# Compare performance for all models
# rf_ho = CSV.read("./rf_all_ho_params.csv", DataFrame)
# gb_ho = CSV.read("./gb_hyperband_rs.csv", DataFrame)
# svm = CSV.read("./svc_hyperband_rs.csv", DataFrame)

perfplot = plot()
for (df, params, name, model) in zip([df_rf, df_gb, df_svm],
                                [rf_params, gb_params, svc_params],
                                ["rf_", "gb_", "svc_",],
                                [rf, gb, svc])
    ho_min = df[argmax(df.accuracy), :]
    ho_min_arr = Array(ho_min)
    # print(ho_min)
    X_train_trans, X_test_trans = preprocessing(
        Dict(:dim_red => dim_reds_dict[Symbol(ho_min[end-1])](ho_min[end-2]),
        :scaler => scalers_dict[Symbol(ho_min[end])]),
        X_train, Y_train
    )
    Random.seed!(0)
    cv_res = crossvalidate(model(ho_min_arr[2:end-3]), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
    # writeTestResult("./" * name, params, ["hyperband_rs"], [ho], [cv_res])
    dir = "./" * name
    CSV.write(dir * "all_cv.csv", DataFrame([cv_res]))
    CSV.write(dir * "all_ho_params.csv", DataFrame(ho_min[2:end]))
    scores = df.accuracy
    # scores = scores[.!isnan.(scores)]
    plot!(perfplot, scores, label=name[1:end-1], ylabel="Accuracy", legend=true) 
end
plot!(perfplot, legend=true, xlabel="Iterations")
plot!(perfplot, xlims=(0, 100))
savefig(perfplot, "perfplot.png")

default_rf = SKLearner("RandomForestClassifier", random_state=0)
default_gb = SKLearner("GradientBoostingClassifier", random_state=0)
default_svc = SKLearner("SVC", random_state=0)
Random.seed!(0)
crossvalidate(default_rf, X_test, Y_test, "accuracy_score", nfolds=5, verbose=true)
crossvalidate(default_gb, X_test, Y_test, "accuracy_score", nfolds=5, verbose=true)
crossvalidate(default_svc, X_test, Y_test, "accuracy_score", nfolds=5, verbose=true)