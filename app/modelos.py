"""Etapa 4: divisão dos dados, modelos candidatos e espaços de busca de hiperparâmetros."""

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from app.features import colunas_features, criar_preprocessador

SEED = 42
LIMITE_CAUDA_DIAS = 60


def dividir_dados(df, usar_categoria=True):
    """Mesma divisão da etapa 2: 80/20 e treino sem pedidos acima de 60 dias.

    O teste mantém todos os pedidos. Devolve X/y de treino, de teste e o prazo estimado
    do teste (fora de X, só para comparação).
    """
    treino, teste = train_test_split(df, test_size=0.2, random_state=SEED)
    treino = treino[treino["y"] <= LIMITE_CAUDA_DIAS]
    colunas = colunas_features(usar_categoria)
    return (
        treino[colunas],
        treino["y"],
        teste[colunas],
        teste["y"],
        teste["prazo_estimado"],
    )


def _pipeline(modelo, usar_categoria=True):
    # O pré-processador fica dentro do Pipeline: em cada fold da validação cruzada ele é
    # ajustado só com o treino do fold (importante para o TargetEncoder, que usa o y).
    return Pipeline([("prep", criar_preprocessador(usar_categoria)), ("modelo", modelo)])


def definir_buscas(usar_categoria=True, n_iter_ensembles=15):
    """Devolve {nome: (tipo_de_busca, pipeline, espaço_de_busca)} para os cinco modelos.

    Grade completa (GridSearchCV) nos modelos leves; amostragem aleatória
    (RandomizedSearchCV) nos ensembles, onde a grade seria cara demais.
    """
    return {
        "Ridge": (
            "grid",
            _pipeline(Ridge(), usar_categoria),
            {"modelo__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
        ),
        "Árvore de decisão": (
            "grid",
            _pipeline(DecisionTreeRegressor(random_state=SEED), usar_categoria),
            {
                "modelo__max_depth": [3, 5, 8, 12],
                "modelo__min_samples_leaf": [10, 50, 200],
            },
        ),
        "KNN": (
            "grid",
            _pipeline(KNeighborsRegressor(), usar_categoria),
            {
                "modelo__n_neighbors": [10, 25, 50, 100],
                "modelo__weights": ["uniform", "distance"],
            },
        ),
        "Random Forest": (
            "random",
            _pipeline(RandomForestRegressor(random_state=SEED), usar_categoria),
            {
                "modelo__n_estimators": randint(100, 300),
                "modelo__max_depth": randint(6, 20),
                "modelo__min_samples_leaf": randint(5, 60),
                "modelo__max_features": [0.3, 0.5, 0.8],
                "modelo__max_samples": [0.5, 0.8],
            },
        ),
        "Gradient Boosting (HGB)": (
            "random",
            _pipeline(HistGradientBoostingRegressor(random_state=SEED), usar_categoria),
            {
                "modelo__learning_rate": loguniform(0.02, 0.3),
                "modelo__max_iter": randint(100, 500),
                "modelo__max_leaf_nodes": randint(8, 64),
                "modelo__min_samples_leaf": randint(10, 200),
                "modelo__l2_regularization": loguniform(1e-3, 10),
            },
        ),
    }


def executar_busca(tipo, pipeline, espaco, X, y, cv=5, n_iter=15):
    """Roda a busca de hiperparâmetros com validação cruzada e devolve o objeto ajustado.

    A métrica de seleção é o RMSE (negativo, convenção do scikit-learn), a mesma que o
    projeto usa para comparar modelos.
    """
    validacao = KFold(cv, shuffle=True, random_state=SEED)
    parametros = dict(
        scoring="neg_root_mean_squared_error",
        cv=validacao,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    if tipo == "grid":
        busca = GridSearchCV(pipeline, espaco, **parametros)
    else:
        busca = RandomizedSearchCV(
            pipeline, espaco, n_iter=n_iter, random_state=SEED, **parametros
        )
    return busca.fit(X, y)


def metricas(y_real, previsto, nome, recorte="todos"):
    """MAE, RMSE e R² de um conjunto de previsões."""
    return {
        "modelo": nome,
        "recorte": recorte,
        "MAE": mean_absolute_error(y_real, previsto),
        "RMSE": root_mean_squared_error(y_real, previsto),
        "R²": r2_score(y_real, previsto),
    }


def avaliar_no_teste(nome, previsto, y_teste):
    """Métricas no teste completo e só nos pedidos de até 60 dias."""
    y_teste = np.asarray(y_teste)
    previsto = np.asarray(previsto)
    ate_limite = y_teste <= LIMITE_CAUDA_DIAS
    return [
        metricas(y_teste, previsto, nome, "todos os pedidos"),
        metricas(y_teste[ate_limite], previsto[ate_limite], nome, f"até {LIMITE_CAUDA_DIAS} dias"),
    ]


def resumo_cv(busca):
    """Tabela com as combinações testadas, ordenada pelo RMSE de validação."""
    r = pd.DataFrame(busca.cv_results_)
    r["RMSE_val"] = -r["mean_test_score"]
    r["RMSE_treino"] = -r["mean_train_score"]
    colunas = ["RMSE_val", "std_test_score", "RMSE_treino"] + [
        c for c in r.columns if c.startswith("param_")
    ]
    return r.sort_values("RMSE_val")[colunas].rename(columns={"std_test_score": "desvio_val"})
