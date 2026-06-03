"""Executa tuning de hiperparametros para datasets/modelos selecionados.

Uso:

    # todos os datasets e modelos
    python -m src.pipeline.run_tune

    # datasets especificos
    python -m src.pipeline.run_tune --task-ids 11 12

    # modelos especificos
    python -m src.pipeline.run_tune --models tabpfn xgboost

    # ambos
    python -m src.pipeline.run_tune \
        --task-ids 11 12 \
        --models tabpfn lightgbm

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sys
from pathlib import Path

# Sobe duas pastas a partir do run_tune.py:
# 1. pipeline -> src
# 2. src -> machine_learning_2026_leandro (raiz)
caminho_raiz = str(Path(__file__).resolve().parents[2])

# Adiciona a pasta raiz ao sistema de buscas do Python
if caminho_raiz not in sys.path:
    sys.path.append(caminho_raiz)
    
from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task

from src.pipeline.split import stratified_split
from src.pipeline.tune import tune

from src.models.run_tune_registry import RUN_TUNE_REGISTRY

DEFAULT_MODELS = list(RUN_TUNE_REGISTRY.keys())

DEFAULT_N_TRIALS = 50
DEFAULT_CV_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tuning.csv"),
        help="CSV de saida com os melhores parametros",
    )

    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="lista opcional de task IDs",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        choices=DEFAULT_MODELS,
        help="lista opcional de modelos",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids is not None else RECOMMENDED_TASK_IDS

    model_names = args.models if args.models is not None else DEFAULT_MODELS

    rows: list[dict] = []

    for task_id in task_ids:
        ds = load_task(task_id)

        X_train, _, y_train, _ = stratified_split(
            ds.X,
            ds.y,
            seed=args.seed,
        )

        print(f"\n=== DATASET: {ds.name} ({task_id}) ===")

        for model_name in model_names:
            print(f"\nTuning {model_name}...")

            config = RUN_TUNE_REGISTRY[model_name]

            best_params, best_score = tune(
                estimator_factory=lambda params: config["factory"](
                    params,
                    seed=args.seed,
                ),
                search_space=config["search_space"],
                X=X_train,
                y=y_train,
                seed=args.seed,
                n_trials=args.n_trials,
                cv_folds=args.cv_folds,
            )

            row = {
                "task_id": task_id,
                "dataset": ds.name,
                "model": model_name,
                "best_score": best_score,
                **best_params,
            }

            rows.append(row)

            print(f"Best score: {best_score:.4f}")
            print(f"Best params: {best_params}")

    pd.DataFrame(rows).to_csv(args.output, index=False)

    print(f"\nResultados gravados em: {args.output}")


if __name__ == "__main__":
    main()
