# Arquitetura do MVP 1

## Objetivo

Transformar vídeo de câmera traseira em uma linha do tempo confiável das posições dos quatro atletas e em métricas físicas e espaciais das duas duplas.

## Pipeline

1. Ingestão de vídeo e metadados.
2. Calibração manual da quadra.
3. Detecção de pessoas.
4. Seleção dos quatro atletas válidos.
5. Tracking temporal.
6. Reidentificação independente dos IDs brutos do detector.
7. Projeção dos pés para coordenadas da quadra em metros.
8. Validação, suavização e interpolação das trajetórias.
9. Cálculo das métricas individuais e coletivas.
10. Geração de vídeo anotado, dados estruturados e relatório.

## Identidades oficiais

- NEAR_LEFT
- NEAR_RIGHT
- FAR_LEFT
- FAR_RIGHT

Os IDs emitidos pelo detector são pistas temporárias e nunca substituem a identidade oficial.

## Qualidade por frame

Cada observação deve registrar confiança de detecção, identidade, projeção e trajetória. Frames suspeitos não podem alimentar aceleração máxima ou outras métricas sensíveis sem validação.

## Primeiras métricas

### Individuais

- distância durante rally;
- velocidade média e máxima;
- aceleração e desaceleração;
- permanência por zona 3x3;
- heatmap;
- posição média.

### Dupla

- centroide;
- distância lateral;
- diferença de profundidade;
- abertura total;
- alinhamento;
- avanço e recuo conjunto;
- tempo de recomposição.

## Fora do MVP 1

- classificação detalhada de golpes;
- diagnóstico técnico;
- recomendações de treino;
- leitura automática de placar.

Esses módulos entram depois que a trajetória dos quatro atletas estiver validada.
