# OList

A tarefa consiste em prever quanto tempo levará para um pedido chegar à casa do cliente, a partir das características do pedido, extraídas das tabelas disponíveis no dataset.

## Contexto

A Olist é uma empresa brasileira de tecnologia voltada ao comércio eletrônico (e-commerce), que ajuda lojistas a venderem seus produtos pela internet, conectando vendedores a grandes marketplaces.

### Problema

Quando um consumidor realiza uma compra, existe uma operação logística para que o produto saia do vendedor e chegue até a residência do cliente. Essa operação envolve diferentes fatores, como localização do vendedor, localização do comprador, características do pedido e transporte.

O projeto utiliza dados históricos desse processo para desenvolver um modelo de aprendizado de máquina capaz de prever o tempo necessário para entregar um pedido.

### Principais tabelas utilizadas

- **olist_orders_dataset.csv**: Contém os dados dos pedidos feitos pelos clientes.
- **olist_customers_dataset.csv**: Contém os dados dos clientes como endereço deles.
- **olist_order_items_dataset.csv**: Contém os dados dos itens vendidos com a informação de qual é vendedor.
- **olist_sellers_dataset.csv**: Contém os dados dos vendedores como endereço deles.
- **olist_geolocation_dataset.csv**: Contém os dados de geolocalização dos endereço baseado nos prefixos de CEP.

### Qual a importância de resolver esse problema?

Uma previsão mais precisa do tempo de entrega pode trazer benefícios como:

- **Planejamento logístico:** melhorar o planejamento das operações de transporte.
- **Experiência do consumidor**: fornecer estimativas de entrega mais realistas.
- **Redução de atrasos**: identificar pedidos com maior risco de ultrapassar o prazo esperado.
- **Tomada de decisão**: identificar fatores que influenciam o tempo de entrega.

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
python data.py
```
