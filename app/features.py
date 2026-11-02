from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ARQUIVOS = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

COLUNAS_DATA = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Retângulo que contém o território brasileiro.
LAT_MIN, LAT_MAX = -33.8, 5.3
LNG_MIN, LNG_MAX = -73.9, -34.7

RAIO_TERRA_KM = 6371

# Grupos de features usados no ColumnTransformer.
COLUNAS_ASSIMETRICAS = ["frete_total", "preco_total", "peso_total_g", "volume_max_cm3"]
COLUNAS_NUMERICAS = [
    "dist_km",
    "n_itens",
    "n_vendedores",
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
    "mes",
    "dia_semana",
    "hora",
]
COLUNAS_BINARIAS = ["mesmo_estado"]
COLUNAS_CATEGORICAS = ["customer_state", "seller_state", "categoria"]


def carregar_tabelas(data_dir):
    """Lê os CSVs usados no projeto e devolve um dicionário {nome: DataFrame}."""
    data_dir = Path(data_dir)
    tabelas = {}
    for nome, arquivo in ARQUIVOS.items():
        tabelas[nome] = pd.read_csv(data_dir / arquivo)
    for coluna in COLUNAS_DATA:
        tabelas["orders"][coluna] = pd.to_datetime(tabelas["orders"][coluna])
    return tabelas


def resumir_geolocalizacao(geolocation):
    """Remove pontos fora do Brasil e resume a geolocalização por prefixo de CEP e por estado.

    O resumo por CEP (mediana) é o que permite juntar sem multiplicar pedidos;
    o resumo por estado serve para imputar CEPs sem geolocalização.
    """
    lat = geolocation["geolocation_lat"]
    lng = geolocation["geolocation_lng"]
    dentro_do_brasil = (lat.between(LAT_MIN, LAT_MAX)) & (lng.between(LNG_MIN, LNG_MAX))
    geo = geolocation[dentro_do_brasil]

    por_cep = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "cep_prefixo"})
    )
    por_estado = (
        geo.groupby("geolocation_state")
        .agg(
            lat_estado=("geolocation_lat", "median"),
            lng_estado=("geolocation_lng", "median"),
        )
        .reset_index()
        .rename(columns={"geolocation_state": "estado"})
    )
    return por_cep, por_estado


def adicionar_coordenadas(tabela, coluna_cep, coluna_estado, por_cep, por_estado):
    """Junta latitude/longitude pelo prefixo do CEP; sem CEP na base, usa a mediana do estado."""
    tabela = tabela.merge(
        por_cep,
        left_on=coluna_cep,
        right_on="cep_prefixo",
        how="left",
        validate="many_to_one",
    ).drop(columns="cep_prefixo")
    tabela = tabela.merge(
        por_estado,
        left_on=coluna_estado,
        right_on="estado",
        how="left",
        validate="many_to_one",
    ).drop(columns="estado")

    sem_coordenada = tabela["lat"].isna()
    tabela["coord_imputada"] = sem_coordenada.astype(int)
    tabela["lat"] = tabela["lat"].fillna(tabela["lat_estado"])
    tabela["lng"] = tabela["lng"].fillna(tabela["lng_estado"])
    return tabela.drop(columns=["lat_estado", "lng_estado"])


def distancia_haversine(lat1, lng1, lat2, lng2):
    """Distância em km entre dois pontos da superfície da Terra (vetorizada)."""
    lat1, lng1, lat2, lng2 = (np.radians(v) for v in (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def construir_dataset(tabelas):
    """Devolve um DataFrame com uma linha por pedido entregue, com `y`, `prazo_estimado` e as features."""
    orders = tabelas["orders"]
    por_cep, por_estado = resumir_geolocalizacao(tabelas["geolocation"])

    # Filtro: só pedidos entregues e com data de entrega preenchida.
    entregue = orders["order_status"] == "delivered"
    com_data_entrega = orders["order_delivered_customer_date"].notna()
    pedidos = orders[(entregue) & (com_data_entrega)].copy()

    # Alvo e prazo estimado em dias fracionados (o prazo estimado NÃO é feature).
    compra = pedidos["order_purchase_timestamp"]
    pedidos["y"] = (
        pedidos["order_delivered_customer_date"] - compra
    ).dt.total_seconds() / 86400
    pedidos["prazo_estimado"] = (
        pedidos["order_estimated_delivery_date"] - compra
    ).dt.total_seconds() / 86400

    # Calendário da compra.
    pedidos["mes"] = compra.dt.month
    pedidos["dia_semana"] = compra.dt.dayofweek
    pedidos["hora"] = compra.dt.hour

    # Clientes com coordenadas.
    clientes = adicionar_coordenadas(
        tabelas["customers"],
        "customer_zip_code_prefix",
        "customer_state",
        por_cep,
        por_estado,
    ).rename(
        columns={
            "lat": "customer_lat",
            "lng": "customer_lng",
            "coord_imputada": "coord_cliente_imputada",
        }
    )
    clientes = clientes[
        [
            "customer_id",
            "customer_state",
            "customer_lat",
            "customer_lng",
            "coord_cliente_imputada",
        ]
    ]
    pedidos = pedidos.merge(
        clientes, on="customer_id", how="left", validate="one_to_one"
    )

    # Vendedores com coordenadas.
    vendedores = adicionar_coordenadas(
        tabelas["sellers"],
        "seller_zip_code_prefix",
        "seller_state",
        por_cep,
        por_estado,
    ).rename(
        columns={
            "lat": "seller_lat",
            "lng": "seller_lng",
            "coord_imputada": "coord_vendedor_imputada",
        }
    )
    vendedores = vendedores[
        [
            "seller_id",
            "seller_state",
            "seller_lat",
            "seller_lng",
            "coord_vendedor_imputada",
        ]
    ]

    # Produtos: peso zero é erro de cadastro, vira ausente.
    produtos = tabelas["products"].copy()
    peso_zero = produtos["product_weight_g"] == 0
    produtos.loc[peso_zero, "product_weight_g"] = np.nan
    produtos["volume_cm3"] = (
        produtos["product_length_cm"]
        * produtos["product_height_cm"]
        * produtos["product_width_cm"]
    )
    produtos = produtos[
        ["product_id", "product_category_name", "product_weight_g", "volume_cm3"]
    ]

    # Itens com produto e vendedor (cada item tem exatamente um produto e um vendedor).
    itens = tabelas["order_items"][
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ]
    ]
    itens = itens[itens["order_id"].isin(pedidos["order_id"])]
    itens = itens.merge(produtos, on="product_id", how="left", validate="many_to_one")
    itens = itens.merge(vendedores, on="seller_id", how="left", validate="many_to_one")

    # Agregação por pedido.
    agregado = itens.groupby("order_id").agg(
        n_itens=("order_item_id", "count"),
        n_vendedores=("seller_id", "nunique"),
        preco_total=("price", "sum"),
        frete_total=("freight_value", "sum"),
        volume_max_cm3=("volume_cm3", "max"),
    )
    # min_count=1: pedido sem nenhum peso conhecido fica ausente em vez de 0.
    agregado["peso_total_g"] = itens.groupby("order_id")["product_weight_g"].sum(
        min_count=1
    )

    # Coordenadas do vendedor: média entre os vendedores distintos do pedido.
    vendedores_do_pedido = itens.drop_duplicates(["order_id", "seller_id"])
    coords_vendedor = vendedores_do_pedido.groupby("order_id").agg(
        seller_lat=("seller_lat", "mean"),
        seller_lng=("seller_lng", "mean"),
        coord_vendedor_imputada=("coord_vendedor_imputada", "max"),
    )

    # Categoria e estado do vendedor vêm do primeiro item do pedido.
    primeiro_item = itens[itens["order_item_id"] == 1].set_index("order_id")
    primeiro_item = primeiro_item[["product_category_name", "seller_state"]].rename(
        columns={"product_category_name": "categoria"}
    )

    agregado = agregado.join(coords_vendedor).join(primeiro_item).reset_index()
    pedidos = pedidos.merge(agregado, on="order_id", how="left", validate="one_to_one")

    pedidos["categoria"] = pedidos["categoria"].fillna("desconhecida")
    pedidos["dist_km"] = distancia_haversine(
        pedidos["customer_lat"],
        pedidos["customer_lng"],
        pedidos["seller_lat"],
        pedidos["seller_lng"],
    )
    pedidos["mesmo_estado"] = (
        pedidos["customer_state"] == pedidos["seller_state"]
    ).astype(int)
    pedidos["coord_imputada"] = (
        (pedidos["coord_cliente_imputada"] == 1)
        | (pedidos["coord_vendedor_imputada"] == 1)
    ).astype(int)

    colunas = (
        ["order_id", "order_purchase_timestamp", "y", "prazo_estimado"]
        + COLUNAS_ASSIMETRICAS
        + COLUNAS_NUMERICAS
        + COLUNAS_BINARIAS
        + COLUNAS_CATEGORICAS
        + ["coord_cliente_imputada", "coord_vendedor_imputada", "coord_imputada"]
    )
    return pedidos[colunas].reset_index(drop=True)


def colunas_features(usar_categoria=True):
    """Lista das colunas que entram em X."""
    categoricas = (
        COLUNAS_CATEGORICAS
        if usar_categoria
        else [c for c in COLUNAS_CATEGORICAS if c != "categoria"]
    )
    return COLUNAS_ASSIMETRICAS + COLUNAS_NUMERICAS + COLUNAS_BINARIAS + categoricas


def criar_preprocessador(usar_categoria=True):
    """ColumnTransformer da etapa 2."""
    categoricas = (
        COLUNAS_CATEGORICAS
        if usar_categoria
        else [c for c in COLUNAS_CATEGORICAS if c != "categoria"]
    )

    assimetricas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("padronizar", StandardScaler()),
        ]
    )
    numericas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("padronizar", StandardScaler()),
        ]
    )
    categoricas_pipe = OneHotEncoder(
        handle_unknown="ignore", min_frequency=100, sparse_output=False
    )

    return ColumnTransformer(
        [
            ("assimetricas", assimetricas, COLUNAS_ASSIMETRICAS),
            ("numericas", numericas, COLUNAS_NUMERICAS),
            ("binarias", "passthrough", COLUNAS_BINARIAS),
            ("categoricas", categoricas_pipe, categoricas),
        ]
    )
