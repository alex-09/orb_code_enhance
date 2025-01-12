from ofs import find_matches
import cv2

# codebase: https://github.com/bayesian-optimization/BayesianOptimization/tree/master/bayes_opt
# user-friendly docs: https://bayesian-optimization.github.io/BayesianOptimization/1.5.1/
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern

import bayes_opt
print("Bayes Opt Version", bayes_opt.__version__)

def optimize_nfeatures(query_image, query_filename, test_images, estimator, preprocess_img, filter_outlier, fixed_nf, algo_name):
    def objective(nfeatures):
        matches_info = find_matches(query_image, query_filename, test_images, nfeatures, estimator, preprocess_img, filter_outlier, fixed_nf, algo_name)
        if matches_info and len(matches_info[0]) > 0:
            # NOTE: matches_info[0][0] = (query_filename, test_filename, inliers_count, nfeatures, total matches)
            _, _, inliers_count, _, total_matches = matches_info[0][0]
            # print("inliers count and TOTAL MATCHES: ", inliers_count, " / ", total_matches)
            return (inliers_count/total_matches) * 100 # the good matches percentage (GMP)
        else:
            return 0

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"nfeatures": (500, 100000)},
        random_state=1,
        verbose=0
    )

    # Possible bounds
    # (500, 5000)
    # (1000, 100000) OUR DEFAULT

    # Hyperparameter needs tuning (set_gp_params: for Gaussian Process Regressor?)
    # optimizer.set_gp_params(kernel=Matern(length_scale=1, nu=1.5), alpha=1e-5)
    # utility = UtilityFunction(kind="ucb", kappa=5, xi=0.0)

    optimizer.maximize(
        init_points=2, 
        n_iter=5, # edit this for to increase or decrease the number of iterations
        # add this parameter if some exception starts to occur 
        # acquisition_function=utility
    )

    # print("OPTIMIZER MAX\n", optimizer.max)
    # sample output {'target': 127.0, 'params': {'nfeatures': 7325.478905769869}}

    return optimizer.max['params']['nfeatures']
