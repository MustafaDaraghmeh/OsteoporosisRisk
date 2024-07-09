import pandas as pd

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("_R01_OsteoporosisRisk.log", mode='w'),
                        logging.StreamHandler()
                    ]
                    )

import matplotlib

matplotlib.use('Agg')


def create_directory(path):
    # Check if the directory already exists
    import os
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
        print(f"Directory '{path}' created")
        return path
    else:
        print(f"Directory '{path}' already exists")
        return path


def get_model_acronym(model_index):
    # Dictionary mapping model indices to their acronyms
    model_acronyms = {
        'xgboost': 'XGB',
        'gbc': 'GBC',
        'lightgbm': 'LGBM',
        'catboost': 'CB',
        'et': 'ET',
        'rf': 'RF',
        'lr': 'LR',
        'lda': 'LDA',
        'qda': 'QDA',
        'dt': 'DT',
        'knn': 'KNN',
        'ada': 'ADA',
        'nb': 'NB',
        'ridge': 'RIDGE',
        'svm': 'SVM'
    }

    # Return the acronym or a default message if the model index is not found
    return model_acronyms.get(model_index, "Unknown Model")


def plot_model(fix_imbalance, X_train, y_train, X_test, y_test, model, figs_path, model_index, title):
    logging.info(f'Plotting {model_index}')

    if model_index=='CB':
        from yellowbrick.contrib.wrapper import wrap, CLASSIFIER
        _model = wrap(model, CLASSIFIER)
    else:
        # To copy the original model
        from sklearn.base import clone
        _model = clone(model)

    # Set up the default plotting settings
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    sns.set_theme(context="paper", style='whitegrid', palette='deep', font_scale=1.6)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    # plt.rcParams['font.size'] = 10  # You can change the base font size here
    mpl.use('Agg')
    # mpl.rcParams['savefig.dpi'] = 300

    try:
        from yellowbrick.classifier import ClassificationReport
        visualizer = ClassificationReport(_model, support=True, title=title, cmap='RdYlGn')
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{fix_imbalance}_{model_index}_ClassificationReport.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ClassificationReport for {model_index} with fix_imbalance={fix_imbalance}")

    try:
        from yellowbrick.classifier import ROCAUC
        visualizer = ROCAUC(_model, title=title, binary=True)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{fix_imbalance}_{model_index}_ROCAUC.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ROCAUC for {model_index} with fix_imbalance={fix_imbalance}")

    try:
        from yellowbrick.classifier import PrecisionRecallCurve
        visualizer = PrecisionRecallCurve(_model, title=title,
                                          per_class=False)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{fix_imbalance}_{model_index}_PrecisionRecallCurve.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: PrecisionRecallCurve for {model_index} with fix_imbalance={fix_imbalance}")

    try:
        from yellowbrick.classifier import ConfusionMatrix
        visualizer = ConfusionMatrix(_model, title=title, cmap='RdYlGn')
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{fix_imbalance}_{model_index}_ConfusionMatrix.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ConfusionMatrix for {model_index} with fix_imbalance={fix_imbalance}")

    try:
        from yellowbrick.classifier import ClassPredictionError
        visualizer = ClassPredictionError(_model, title=title)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{fix_imbalance}_{model_index}_ClassPredictionError.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ClassPredictionError for {model_index} with fix_imbalance={fix_imbalance}")

    pass


def model_index_export_tables(df: pd.DataFrame, csv_path: str, tex_path: str, label: str, caption: str):
    df = df.copy()
    df.index.rename('Model', inplace=True)
    df.reset_index(drop=False, inplace=True)
    df['Model'] = df['Model'].apply(lambda x: get_model_acronym(x))
    df.to_csv(f'{csv_path}/{label}.csv', index=False)
    df.to_latex(f'{tex_path}/{label}.tex', index=False, float_format="%.4f",
                label=f'tbl:{label}',
                caption=caption)
    pass


def run_classification_pipeline(dataset, experiment_name=None, random_state=123, fix_imbalance=False, fix_imbalance_method=None):
    logging.info(f'Classification pipeline is started ({experiment_name})')

    # Experiment Short Name (ESN)
    esn = ''.join([w[0] for w in experiment_name.split()])

    # add the random_state to the ESN
    esn = f'{random_state}_' + esn

    # Configure the experiment setups and data preparation processes involved in this stage
    from pycaret.classification import ClassificationExperiment
    cla = ClassificationExperiment()
    cla.setup(data=dataset, train_size=0.7, target='OP',
              fold_strategy="stratifiedkfold", fold=10,
              normalize=True, normalize_method='zscore',  # minmax, zscore
              low_variance_threshold=0,

              fix_imbalance=fix_imbalance, fix_imbalance_method=fix_imbalance_method,
              #
              # remove_outliers = False,
              # outliers_method  = "iforest",
              # outliers_threshold  = 0.05,
              #
              # transformation=False, transformation_method="yeo-johnson",
              #
              # remove_multicollinearity=False, multicollinearity_threshold=0.9,
              # pca=False, pca_method="linear", pca_components=0.99,  # 'mle' only for pca_method='linear'

              system_log=True, experiment_name=f'{random_state} - ' + experiment_name,
              session_id=random_state, n_jobs=-1,
              memory=f"./tmp_{esn}",
              html=False
              )

    # ********************************************************
    # logging.info("Plot the Anomalies subject to first PCA component.")
    # Path to save the figures
    figs_path = create_directory(path=f'./{esn}/figs')

    # ********************************************************
    # To determine the type of the classification problem
    classification_type = 'binary' if not cla.get_config('is_multiclass') else 'multiclass'

    # Removes Kappa and MCC metrics from the experiment results.
    cla.remove_metric('Kappa')
    cla.remove_metric('MCC')

    # Show the target class distribution on both train and test datasets
    logging.info(f'Class distribution after the data preparation stage:'
                 f'\nTrain: \n{cla.y_train_transformed.value_counts()}'
                 f'\nTest: \n{cla.y_test_transformed.value_counts()}'
                 )

    # *******************************************************************************************
    # Stage 01: Compare a baseline models.
    # The models of this stage are evaluated on both CV and hold out data, including various plots.
    # *******************************************************************************************
    logging.info(f"Start the Stage 01 of the Experiment")
    logging.info(f"Compare a baseline models on CV data, sorted by F1")

    # Candidate model indexes
    candidate_models = ['xgboost', 'gbc', 'lightgbm', 'catboost', 'et', 'rf', 'lr', 'lda', 'qda', 'dt', 'knn', 'ada',
                        'nb', 'ridge', "svm"]

    # m_select represents the number of model subject for the calibration process. In this experiment, we consider all the candidate models
    m_select = len(candidate_models)

    # Run the comparison process
    top_models = cla.compare_models(n_select=m_select, sort="f1", include=candidate_models)

    # Get the comparison results of the candidate models using the CV data, sorted by F1
    cv_res_compare_models_df = cla.pull()
    cv_res_compare_models_df.drop(['Model', 'TT (Sec)'], axis=1, inplace=True)

    # In the following lines of codes, we examine using hold out data, calibrate the models, examining the calibrated models on CV and holdout data, and determine the best performing one for each calibrated method, then plot the possible figures.
    print("\nExamine the models on CV and holdout data, and determine the best performing one.")

    # To hold the candidate models results on hold holdout data
    ho_res_top_models = []

    logging.info("Iterate over the candidate models.")
    # Iterate over the candidate models. Note that the models are already created and evaluated using CV data.
    for estimator in top_models:
        # --------------------------------------------------------
        # Step 01: Evaluate the candidate models on hold out data.
        # --------------------------------------------------------

        # Print the model name and its parameter
        logging.info(f"Start working with {cla._get_model_name(estimator)}")
        logging.info(f"Params:\n {estimator.get_params(True)}")
        logging.info("The results using the holdout Data:")

        # Predict using the holdout data
        cla.predict_model(estimator=estimator)

        # Returns the latest results and save them
        ho_res_top_models.append(cla.pull())

        # Plot the model. Note that some models not support all the figure.
        model_acronym = get_model_acronym(cla._get_model_id(estimator))
        plot_model(fix_imbalance, cla.X_train_transformed, cla.y_train_transformed, cla.X_test_transformed,
                   cla.y_test_transformed, estimator, figs_path,
                   model_acronym,
                   f"Model: {model_acronym}     Fix Imbalance={fix_imbalance}")

        logging.info(f"End working with {cla._get_model_name(estimator)}")

    # ------------------------------------
    # Export the results of stage 01
    # ------------------------------------

    print('*' * 50)
    print('Export the results of the candidate estimators')
    csv_path = create_directory(path=f'./{esn}/tables/csv')
    tex_path = create_directory(path=f'./{esn}/tables/tex')

    print('*' * 50)
    # Export the results of the classifiers on CV data

    print(f"Comparative performance of {classification_type} classifiers trained {'with' if fix_imbalance else 'without'} the application of fixed imbalance using {fix_imbalance_method} on cross-validation data, sorted by F1-score.")
    print(cv_res_compare_models_df)
    model_index_export_tables(cv_res_compare_models_df, csv_path, tex_path,
                              label=f"{classification_type}_{fix_imbalance}_{fix_imbalance_method}_cv_res_compare_models_df",
                              caption=f"Comparative performance of {classification_type} classifiers trained {'with' if fix_imbalance else 'without'} the application of fixed imbalance using {fix_imbalance_method} on cross-validation data, sorted by F1-score."
                              )

    # Export the results of the classifiers without calibration process on holdout data
    print(f"Comparative performance of {classification_type} classifiers trained {'with' if fix_imbalance else 'without'} the application of fixed imbalance using {fix_imbalance_method} on holdout data, sorted by F1-score.")
    ho_res_top_models_df = pd.concat(ho_res_top_models, axis=0, ignore_index=True)
    ho_res_top_models_df.index = cv_res_compare_models_df.index[:m_select]
    ho_res_top_models_df.drop(['Model'], axis=1, inplace=True)
    ho_res_top_models_df.sort_values(by='F1', ascending=False, inplace=True)
    print(ho_res_top_models_df)
    model_index_export_tables(ho_res_top_models_df, csv_path, tex_path,
                              label=f"{classification_type}_{fix_imbalance}_{fix_imbalance_method}_ho_res_top_models_df",
                              caption=f"Comparative performance of {classification_type} classifiers trained {'with' if fix_imbalance else 'without'} the application of fixed imbalance using {fix_imbalance_method} on holdout data, sorted by F1-score."
                              )

    print('*' * 50)
    logging.info(f"Stage 01 of the experiment is completed")
    # ---------------------------------------------------------------


    logging.info(f'Classification pipeline is Completed ({experiment_name})\n')
    pass

def main():
    random_state = 0
    df = pd.read_csv('UA.csv')

    # Run the classification pipeline in the binary dataset
    run_classification_pipeline(dataset=df,
                                experiment_name='Binary - Osteoporosis Risk',
                                fix_imbalance=False,
                                fix_imbalance_method='SMOTE',
                                random_state=random_state)

    # Run the classification pipeline in the binary dataset
    run_classification_pipeline(dataset=df,
                                experiment_name='Binary SMOTE - Osteoporosis Risk',
                                fix_imbalance=True,
                                fix_imbalance_method='SMOTE',
                                random_state=random_state)

    print('Experiment completed')

if __name__ == '__main__':
    main()
