# Álgebra Linear — Multiplicação de Matrizes Densas (GEMM)

Este repositório compara o desempenho do algoritmo GEMM utilizando diferentes paradigmas de processamento paralelo e otimização de memória.

## Abordagens Implementadas
* **CPU Sequencial:** Resolução algorítmica básica de complexidade $O(N^3)$.
* **CUDA Naive:** Paralelização mapeando 1 thread para cada elemento da matriz na Memória Global.
* **CUDA Tiled:** Otimização utilizando **Shared Memory** estruturada em blocos de $32 \times 32$ para maximizar o reúso de dados e reduzir o gargalo de largura de banda.

## Como Executar (Google Colab)

Para rodar os testes utilizando uma GPU NVIDIA disponibilizada na nuvem:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guidias-sketch/Algebra-linear-multiplicacao-de-matrizes-densas-GEMM/blob/main/GEMM_Benchmarks.ipynb)

> **Importante:** Lembre-se de ir em *Ambiente de Execução* -> *Alterar tipo de ambiente de execução* e selecionar a **GPU T4** antes de rodar o script.
