from autogluon.tabular import TabularPredictor
from src.common.tabrepo_2024_custom import zeroshot2024
import hydra
from hydra.utils import instantiate
from loguru import logger

from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import re


@hydra.main(version_base=None, config_path='conf', config_name='catboost_exp')
def main(cfg: DictConfig) -> None:
    train = pd.read_csv(cfg.data.train_path)
    test = pd.read_csv(cfg.data.test_path)
    label = "hard_live_cost"

    logger.info(f"train shape: {train.shape}")
    logger.info(f"test shape: {test.shape}")

    # -- Preprocessing
    train = train.drop(columns=["serial_number"])
    test = test.drop(columns=["serial_number"])

    def update(df, is_train=True):
        t = 10

        cat_c = ['model', 'capacity_bytes']

        for col in cat_c:
            df[col] = df[col].fillna('missing')
            df[col] = df[col].astype('category')

        return df

    train = update(train)
    test = update(test, is_train=False)

    train = train.drop_duplicates()

    allowed_models = [
        "LR",
        "GBM",
        "CAT",
        "XGB",
        "RF",
        "XT",
    ]

    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
    logger.info(f"allowed_models: {allowed_models}")

    # -- Run AutoGluon
    predictor = TabularPredictor(
        label=label,
        eval_metric="rmse",
        problem_type="regression",
        verbosity=2,
    )

    predictor.fit(
        time_limit=int(60 * 60 * 3),
        train_data=train,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        # Early Stopping
        ag_args_fit={
            "stopping_metric": "rmse",
        },
        # Validation Protocol
        num_bag_folds=10,
        num_bag_sets=2,
        num_stack_levels=3,
    )
    logger.info(f"End fit")
    predictor.fit_summary(verbosity=1)
    print(predictor.leaderboard())
    predictions = predictor.predict(test)
    logger.info(f"predictions shape: {predictions.shape}")
    # -- Save Predictions
    submission = pd.read_csv(cfg.data.sample_submission_path)
    submission[label] = predictions
    submission.to_csv(cfg.data.result_path, index=False)
    logger.info(f"saved")


if __name__ == '__main__':
    main()
