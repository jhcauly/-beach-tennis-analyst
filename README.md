# Beach Tennis Analyst

Sistema de visão computacional para análise física, espacial e tática de partidas de beach tennis usando somente câmera.

## Objetivo do MVP 1

Processar um vídeo de câmera traseira, identificar quatro atletas, projetar suas posições em uma quadra 2D e gerar métricas confiáveis de deslocamento e organização das duplas.

## Escopo inicial

- calibração manual da quadra;
- detecção e tracking de quatro jogadores;
- identidade persistente com confiança por frame;
- coordenadas em metros;
- filtragem e validação de trajetórias;
- distância, velocidade e aceleração;
- heatmaps e ocupação 3x3;
- centroide, distância e alinhamento das duplas;
- relatório físico e espacial.

## Princípios

1. Nenhuma métrica sem indicador de confiança.
2. IDs do detector não são identidade oficial do atleta.
3. Frames suspeitos devem ser marcados, não escondidos.
4. O sistema deve separar fatos observados de interpretações táticas.
5. Bola, golpes e diagnóstico entram somente após o tracking espacial estar validado.

## Marco atual

**M1 — Tracking Espacial Confiável**
