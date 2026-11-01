# Olist

A tarefa consiste em prever quanto tempo levará para um pedido chegar à casa do cliente, a partir das características do pedido, extraídas das tabelas disponíveis no dataset.

## Contexto

A Olist é uma empresa brasileira de tecnologia voltada ao comércio eletrônico (e-commerce), que ajuda lojistas a venderem seus produtos pela internet, conectando vendedores a grandes marketplaces.

### Problema

Quando um consumidor realiza uma compra, existe uma operação logística para que o produto saia do vendedor e chegue até a residência do cliente. Essa operação envolve diferentes fatores, como localização do vendedor, localização do comprador, características do pedido e transporte.

O projeto utiliza dados históricos desse processo para desenvolver um modelo de aprendizado de máquina capaz de prever o tempo necessário para entregar um pedido.

### Definição do problema de regressão

- **Amostra:** um pedido entregue (uma linha por `order_id`).
- **Variável alvo:** tempo de entrega, em dias (data de entrega ao cliente − data da compra), em valor fracionado.
- **Momento da previsão:** no ato da compra. Só entram variáveis conhecidas nesse momento; a avaliação do cliente e as datas de despacho ficam de fora para evitar vazamento de dados.
- **Escopo:** pedidos com status `delivered` e data de entrega preenchida (96.470 pedidos).

### Principais tabelas utilizadas

- **orders** (`olist_orders_dataset.csv`): datas que formam o alvo e o vínculo com o cliente.
- **order_items** (`olist_order_items_dataset.csv`): liga o pedido ao produto e ao vendedor, e traz preço e frete.
- **products** (`olist_products_dataset.csv`): peso, dimensões e categoria influenciam o transporte.
- **customers** (`olist_customers_dataset.csv`) e **sellers** (`olist_sellers_dataset.csv`): CEP e estado de destino e de origem.
- **geolocation** (`olist_geolocation_dataset.csv`): latitude e longitude por prefixo de CEP, para calcular a distância entre vendedor e cliente.
- **Não utilizadas:** `order_reviews` (só existe após a entrega, causaria vazamento de dados), `order_payments` (fora de escopo) e `product_category_name_translation` (só traduz a categoria).

### Como as tabelas se relacionam

- `orders` 1—N `order_items`
- `order_items` N—1 `products` e N—1 `sellers`
- `orders` N—1 `customers`
- `customers` e `sellers` se ligam a `geolocation` pelo prefixo do CEP

### Qual a importância de resolver esse problema?

Uma previsão mais precisa do tempo de entrega pode trazer benefícios como:

- **Planejamento logístico:** melhorar o planejamento das operações de transporte.
- **Experiência do consumidor**: fornecer estimativas de entrega mais realistas.
- **Redução de atrasos**: identificar pedidos com maior risco de ultrapassar o prazo esperado.
- **Tomada de decisão**: identificar fatores que influenciam o tempo de entrega.

### Desafio real

Mesmo com tudo o que se sabe no momento da compra (distância entre vendedor e cliente, estados, frete, peso, volume e calendário), um modelo de referência explica só cerca de **36% da variância** do prazo de entrega (R² 0,36; RMSE de 7,5 dias contra 9,3 dias ao prever sempre a média). Os 0,3% de pedidos que levam mais de 60 dias respondem por cerca de **um quarto do erro quadrático total**, e o erro é maior para longas distâncias e para os estados do Norte e do Nordeste. A hipótese é que boa parte do prazo depende de fatores que só acontecem depois da compra — tempo de postagem do vendedor, transportadora e eventos externos —, o que põe um teto no que qualquer modelo consegue com esses dados.

## Instalação

### Linux

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```sh
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

## Execução

```sh
python app/data.py
```
