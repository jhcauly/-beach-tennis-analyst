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

## Análise somente no enquadramento calibrado

O sistema compara cada quadro do vídeo com a imagem usada na calibração antes
de detectar atletas. A câmera lateral, os closes e o zoom são ignorados até
reaparecer o enquadramento compatível com a posição dos quatro cantos marcados.
A posição exata dos atletas não precisa ser igual à da referência.

A análise começa após três quadros consecutivos compatíveis. Se a transmissão
trocar de câmera, ela fica suspensa e retoma quando a câmera calibrada voltar.
A espera pela câmera correta não consome o limite de tentativas de identificação
dos atletas. As métricas não somam deslocamentos entre os trechos separados
por cortes de câmera. O vídeo de saída preserva a duração e os tempos originais.

Ao salvar uma nova calibração, mantenha juntos o arquivo JSON e a imagem limpa
`<nome>.reference.png` criada ao lado dele. Para calibrações antigas, mantenha a
imagem com o mesmo nome do vídeo na pasta `referencia`, como antes.
Se não houver uma referência válida, o sistema informa o problema em vez de
analisar uma câmera incompatível.

O reconhecimento usa detalhes visuais e a geometria da cena, desconsiderando
o placar no canto superior esquerdo. Mudanças de iluminação moderadas são
toleradas; uma imagem muito desfocada ou sem detalhes pode suspender a análise.
`summary.json` registra quantos quadros foram aceitos e ignorados em
`camera_view_filter`.
