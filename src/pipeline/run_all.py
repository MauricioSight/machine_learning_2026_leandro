"""Executa o pipeline completo: para cada dataset, treina e avalia todos os modelos.

Uso:
    python -m src.pipeline.run_all --seed 42 --output results/raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import gc
import torch
import numpy as np
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
from src.models.automl import build_autogluon


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

        factories: dict[str, callable] = {}

        # Registra placeholders para o IF interceptar lá dentro do loop
        #factories["autogluon_default"] = lambda seed: None 
        factories["autogluon_extreme"] = lambda seed: None

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
            # --- IDENTIFICA SE É O AUTOGLUON ---
            if "autogluon" in model_name:
                # Determina o preset com base no nome que você escolheu na factory
                preset_type = "extreme" if "extreme" in model_name else "default"
                
                # 1. Chama a sua função original para pegar a tupla (retorna o predictor desconfigurado)
                # Passamos o nome da coluna alvo que vamos criar temporariamente
                target_col = "__target_label__"

                num_classes = len(np.unique(y_train))
                if num_classes == 2:
                    eval_metric = "roc_auc"
                else:
                    # Usa a métrica multiclasse original se for > 2
                    eval_metric = "roc_auc_ovo_macro"
                limite_tempo = 14400 if preset_type == "extreme" else None

                predictor, ag_preset, time_limit = build_autogluon(
                    label=target_col,
                    eval_metric = eval_metric, 
                    seed=args.seed, 
                    preset=preset_type,
                    time_limit_seconds=limite_tempo
                )
                
                # 2. Prepara os dados: Une X_train e y_train em um único DataFrame
                train_data = X_train.copy()
                train_data[target_col] = y_train

                
                # 3. Executa o treinamento nativo do AutoGluon (passando os parâmetros da tupla)
                print(f"[{ds.name}] Iniciando treino do AutoGluon ({preset_type})...")
                
                # Configurando hyperparameters para a v1.4 caso seja o extreme
                hyperparameters = "multimodal" if preset_type == "extreme" else "default"
                
                estimator = predictor.fit(
                    train_data=train_data,
                    presets=ag_preset,
                    time_limit=time_limit,
                    hyperparameters=hyperparameters
                )
                
                # Criamos um "adaptador rápido" em tempo de execução para que o seu 
                # evaluate.py consiga chamar .predict() e .predict_proba() sem quebrar
                class AGAdapter:
                    def __init__(self, pred): self.pred = pred
                    def fit(self, X, y): pass # Já foi treinado acima
                    def predict(self, X): return self.pred.predict(X).values
                    def predict_proba(self, X): return self.pred.predict_proba(X).values
                
                eval_estimator = AGAdapter(estimator)
                
            else:
                # Fluxo normal para LightGBM, XGBoost, CatBoost e o modelo do grupo
                estimator = factory(args.seed)
                eval_estimator = estimator
            metrics = fit_predict_evaluate(eval_estimator, X_train, y_train, X_test, y_test)
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
