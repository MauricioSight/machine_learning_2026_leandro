# Model Card: TabPFN-2.5

## 1. Detalhes do modelo

- **Nome:** TabPFN-2.5
- **Versão:** 2.5
- **Autores originais:** L. Grinsztajn et al., 2026
- **Repositório oficial:** https://github.com/PriorLabs/TabPFN
- **Licença do código:** Apache 2.0
- **Licença dos pesos pré-treinados (se aplicável):** Licença Proprietária/Prior Labs (gratuita para uso não-comercial acadêmico, exigindo registro e geração de token de API)
- **Família arquitetural:** O sistema é um Foundation Model Tabular estruturado através de uma rede Transformer. Opera usando aprendizado em contexto (in-context learning) e mecanismos de atenção alternada (sobre características e amostras).
- **Contagem de parâmetros:** Não está explícito no artigo de base. Todavia, a profundidade arquitetural da rede conta com 18 camadas focadas em regressão e 24 camadas para tarefas de classificação.
- **Complexidade computacional:** A complexidade do algoritmo é expressa matematicamente como $\mathcal{O}(r^{2}\min(c,500)+r\min(c,500)^{2})$, onde $r$ representa o total de linhas da base e $c$ a quantidade de colunas do conjunto.
- **Pico de memória observado:** 16GB durante o processamento do modelo.
- **Toolkit / dependências:** tabpfn>=2.5, torch 2.x, pytabkit (recomendada a utilização de GPU).
- **Hiperparâmetros principais:** Não utilizado pois segundo o site dos desenvolvedores do modelo, o modelo já se adapta no tempo de inferência adicionando falor apenas em alguns cenários específicos

## 2. Uso pretendido

- **Caso de uso primário:** Tarefas baseadas em dados tabulares para regressão e classificação multivariada que exijam capacidade avançada de generalização com calibração confiável numa única passagem direta do modelo. Também demonstra eficiência destacada como meta-learner para estimativas formais em inferência causal.
- **Casos de uso fora de escopo:** Não especificada no artigo de forma estrita. Porém, inferências complexas sem destilação e em proporções excedendo largamente a casa das centenas de milhares de linhas exigem processamentos alternativos não ideais.
- **Usuários pretendidos:** O sistema foi formulado visando atender pesquisadores, desenvolvedores na área médica, cientistas de modelagem ecológica e atuários do setor financeiro que lidam rotineiramente com o problema de escassez crônica de dados.
- **Faixa de n suportada:** Desenvolvido organicamente para atuar com extrema eficiência processando de zero a 50.000 amostras individuais de treinamento. Em benchmarks padronizados obteve resultados estáveis em escalas de até 100.000 registros de treino.
- **Faixa de p suportada:** Adequado para conjuntos estruturados possuindo até o máximo prático de 2.000 propriedades independentes (features) de classificação ou regressão.
- **Condições operacionais:** Requer GPU com pelo menos 16 GB de VRAM para inferência rápida em datasets médios e para evitar gargalos em atenção quadrática.

## 3. Fatores observados

Dimensões em que o desempenho do modelo varia, avaliadas neste projeto sobre os 30 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** Altamente escalável para os regimes médio (1.000 a 10.000) e grande (> 10.000), mantendo liderança. No regime pequeno (< 1.000), o modelo apresenta leve desvantagem frente a ensembles massivos, mas segue competitivo.
- **Número de classes:** Robusto tanto em classificação binária quanto multiclasse, entregando ganhos estáveis em relação aos baselines de árvore em ambos.
- **Proporção entre features categóricas e numéricas:** Demonstrou ser invariante à densidade categórica, garantindo a melhor performance em datasets de alta e baixa proporção sem degradação na codificação.
- **Presença de valores ausentes:** O conjunto de dados foi tratado anteriormente para não apresentar valores ausetes.

## 4. Métricas alcançadas

Tabela agregada nos 30 datasets do TabArena. Reportar média, desvio padrão e intervalo de confiança de 95% via bootstrap (1.000 reamostragens).

| Métrica | Média | Desvio | IC 95% (Não-paramétrico) | Ranking médio |
|---|---|---|---|---|
| AUC OvO | 0,872 | 0,097 | [0,821; 0,922] | 1,47 |
| Accuracy | 0,872 | 0,089 | N/A | N/A |
| G-Mean | 0,663 | 0,291 | N/A | N/A |
| Cross-Entropy | 0,297 | 0,171 | N/A | N/A |
| Tempo total (s) | 529,60 | 981,83 | N/A | N/A |

### Resultados por regime

- **Tamanho:** pequeno: AUC=0,869; médio: AUC=0,882; grande: AUC=0,849
- **Número de classes:** binário: AUC=0,861; multiclasse: AUC=0,914
- **Proporção categórica:** baixa: AUC=0,875; alta: AUC=0,868
- **Missing values:** com NaN: AUC=0,828; sem NaN: AUC=0,883

## 5. Dados de avaliação

- **Origem:** 30 datasets do TabArena-v0.1 (NeurIPS 2025), via OpenML.
- **Distribuição por regime:** 3 pequenos + 19 médios + 8 grandes.
- **Estratégia de split:** 70/30 estratificado por classe, seed=42.
- **Pré-processamento aplicado:** Foi adicionado o valor da mediana em valores nulos numéricos. E em variáveis categóricas foi adicionado o valor 'Missing' antes da conversão em códigos numéricos
- **Lista dos datasets utilizados:** Relação completa e detalhada presente no corpo do Relatório de experimentos do grupo.

## 6. Dados de treino e pré-treino

- **Modelo é foundation model pré-treinado, treinado do zero ou híbrido?** TabPFN-2.5 trata-se essencialmente de um Foundation Model tabular pré-treinado maciçamente.
- **Origem dos dados de pré-treino (se aplicável):** Os conhecimentos prévios e distribuições base provêm integralmente de milhões de distribuições sintéticas que emulam dados tabulares heterogêneos.
- **Origem dos dados de treino direto (se aplicável):** Variante Real-TabPFN-2.5 inclui aperfeiçoamento residual e fine-tuning contínuo derivado de 43 conjuntos puros originários do mundo real curados no ecossistema OpenML e Kaggle.
- **Possíveis vieses herdados do pré-treino:** Assunções e prioris embutidas no gerador de dados sintéticos podem não cobrir correlações espúrias ou distribuições de cauda longa estritas do mundo real.

## 7. Análise quantitativa

- **Posição no ranking médio entre os sistemas avaliados** : 1 de 6.
- **Friedman + Nemenyi:** O modelo rejeita a hipótese nula com superioridade estatisticamente significante contra LightGBM, CatBoost e XGBoost. Foi inserido no mesmo grupo de equivalência estatística (barra do diagrama de diferença crítica) que o AutoGluon Default e Extreme.
- **Bayesian signed-rank com ROPE = 0,01 em AUC:** Atestou-se forte equivalência prática contra AutoGluon Extreme (p_equivalente = 0,992) e AutoGluon Default (p_equivalente = 0,991). O modelo do grupo é claramente melhor que o XGBoost (p_melhor = 1,000), CatBoost (0,997) e LightGBM (0,938).
- **Comparação com AutoGluon:** Atingiu o mesmo teto de performance do preset extreme 4h consumindo apenas ~30% do tempo de processamento. Contra o preset default, apresenta leve incremento de AUC (0,872 vs 0,866) com custo de tempo de 3,7x.
- **Quebra por regime:** O modelo vence rigorosamente os baselines clássicos em regimes médios, grandes, com dados ausentes e invariância categórica, justificando sua arquitetura in-context. Perdeu por margem irrelevante (0,001) apenas no regime pequeno para o AutoGluon Extreme.

## 8. Considerações éticas

- **Riscos de uso indevido:** Decisões opacas em domínios sensíveis devido à interpretabilidade local desafiadora inerente a redes Transformer de bilhões de parâmetros em relação a modelos lineares ou árvores rasas.
- **Fairness por classe:** Risco de underfitting em classes minoritárias com baixo recall caso o dataset real possua topologia severamente divergente da meta-aprendizagem sintética.
- **Dependência de licença de pesos pré-treinados:** TabPFN-2.5 possui licença não-comercial; uso em produção industrial exige avaliação jurídica e licenciamento com a Prior Labs.
- **Impacto ambiental:** Exigência de inferência via GPU consome mais energia e apresenta maior pegada de carbono do que o treinamento rápido via CPU com baselines como LightGBM.
- **Recomendações de auditoria:** Comparar predições locais e distribuições globais com um baseline interpretável, como EBM (Explainable Boosting Machine), antes de validar deploys críticos.

## 9. Avisos e recomendações

- **Quando usar este modelo:** Regimes com bases de dados de tamanho médio, alta incidência de valores ausentes, e quando o limite máximo do desempenho estatístico justifica custos de processamento na ordem de minutos ao invés de segundos.
- **Quando NÃO usar este modelo:** Regimes onde restrições operacionais exijam inferência e latência em tempo real (milissegundos), ausência de GPU no ambiente de deploy, e limitações de licenciamento estritamente open-source comercial.
- **Alternativas recomendadas em cada caso:** Para inferência de baixa latência em tempo real, usar LightGBM; para bases com dominância absoluta de features categóricas sem processamento, usar CatBoost; para AutoML genérico automatizado veloz, usar AutoGluon default.

## 10. Reprodutibilidade

- **Ambiente:** Python 3.11, dependências e bibliotecas fixadas em `pyproject.toml`.
- **Hardware utilizado:** GPU NVIDIA GeForce RTX 5060 ti 16GB, 32GB de RAM, com tempo total médio de 529,60s por inferência agregada.
- **Comandos para reproduzir:**
  ```bash
  uv sync
  python -m src.pipeline.run_all-include-group-model --seed 42
- **Hash do commit:** 0b6b9f5dbf92d07f2010bcae8491d1041173ac6b
