"""Executa o pipeline completo: para cada dataset, treina e avalia todos os modelos.

Uso:
    python -m src.pipeline.run_all --seed 42 --output results/raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import gc
import torch

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
from src.models.baselines import BASELINE_FACTORIES
from src.models.group_model import build_group_model
from src.pipeline.evaluate import fit_predict_evaluate
from src.pipeline.split import stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw.csv"),
        help="caminho do CSV de saida",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="opcional: lista de task IDs do OpenML; se omitido, usa RECOMMENDED_TASK_IDS",
    )
    parser.add_argument(
        "--include-group-model",
        action="store_true",
        help="se passado, inclui o modelo do grupo (build_group_model)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids else RECOMMENDED_TASK_IDS

    rows: list[dict] = []
    for task_id in task_ids:
        ds = load_task(task_id)
        X_train, X_test, y_train, y_test = stratified_split(ds.X, ds.y, seed=args.seed)

        factories: dict[str, callable] = dict(BASELINE_FACTORIES)
        if args.include_group_model:
            factories["group_model"] = build_group_model

        header_written = False
        for model_name, factory in factories.items():

            # ==========================================
            # INÍCIO DA LIMPEZA DE MEMÓRIA DA GPU
            # ==========================================
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # ==========================================

            estimator = factory(args.seed)
            metrics = fit_predict_evaluate(estimator, X_train, y_train, X_test, y_test)
            row = {"task_id": task_id, "dataset": ds.name, "model": model_name}
            row.update(metrics.to_dict())
            rows.append(row)
            print(
                f"[{ds.name}] {model_name}: AUC={metrics.auc_ovo:.4f}, "
                f"ACC={metrics.accuracy:.4f}, time={metrics.fit_time_s + metrics.predict_time_s:.1f}s"
            )

            pd.DataFrame([row]).to_csv(
                args.output,
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True

    print(f"\nResultados gravados em {args.output}")


if __name__ == "__main__":
    main()
