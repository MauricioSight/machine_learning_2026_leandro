# Model Card: TabPFN-2.5

> Preencha este template para o modelo principal atribuído ao seu grupo. Substitua os campos `<...>` pelos valores reais. Não deixe campos em branco; use "N/A" quando não aplicável.
>
> Estrutura inspirada em Mitchell et al. (2019), com extensões específicas da disciplina (fatores observados nos quatro regimes do TabArena, análise quantitativa contra baselines e AutoGluon, e seção de avisos e recomendações).

## 1. Detalhes do modelo

- **Nome:**  TabPFN-2.5
- **Versão:** 2.5
- **Autores originais:** L. Grinsztajn et al., 2026
- **Repositório oficial:** (https://github.com/PriorLabs/TabPFN)
- **Licença do código:** ex.: Apache 2.0
- **Licença dos pesos pré-treinados (se aplicável):** Licença Proprietária/Prior Labs (gratuita para uso não-comercial acadêmico, exigindo registro e geração de token de API)
- **Família arquitetural:** O sistema é um Foundation Model Tabular estruturado através de uma rede Transformer. Opera usando aprendizado em contexto (in-context learning) e mecanismos de atenção alternada (sobre características e amostras).
- **Contagem de parâmetros:** Não está explícito no artigo de base. Todavia, a profundidade arquitetural da rede conta com 18 camadas focadas em regressão e 24 camadas para tarefas de classificação
- **Complexidade computacional:** A complexidade do algoritmo é expressa matematicamente como $\mathcal{O}(r^{2}\min(c,500)+r\min(c,500)^{2})$, onde $r$ representa o total de linhas da base e $c$ a quantidade de colunas do conjunto.  
- **Pico de memória observado:** 16Gb durante o processamento do modelo
- **Toolkit / dependências:** tabpfn>=2.5, torch 2.x, recomendado a utilização de GPU
- **Hiperparâmetros principais:** <listar; indicar se foi feita busca via Optuna>

## 2. Uso pretendido

- **Caso de uso primário:** Tarefas baseadas em dados tabulares para regressão e classificação multivariada que exijam capacidade avançada de generalização com calibração confiável numa única passagem direta do modelo. Também demonstra eficiência destacada como meta-learner para estimativas formais em inferência causal.
- **Casos de uso fora de escopo:** Não especificada no artigo de forma estrita. Porém, inferências complexas sem destilação e em proporções excedendo largamente a casa das centenas de milhares de linhas exigem processamentos alternativos não ideais.
- **Usuários pretendidos:**O sistema foi formulado visando atender pesquisadores, desenvolvedores na área médica, cientistas de modelagem ecológica e atuários do setor financeiro que lidam rotineiramente com o problema de escassez crônica de dados
- **Faixa de n suportada:** Desenvolvido organicamente para atuar com extrema eficiência processando de zero a 50.000 amostras individuais de treinamento. Em benchmarks padronizados obteve resultados estáveis em escalas de até 100.000 registros de treino.
- **Faixa de p suportada:** Adequado para conjuntos estruturados possuindo até o máximo prático de 2.000 propriedades independentes (features) de classificação ou regressão.
- **Condições operacionais:** <ex.: requer GPU com pelo menos 8 GB de VRAM para inferência rápida em datasets médios>

## 3. Fatores observados

Dimensões em que o desempenho do modelo varia, avaliadas neste projeto sobre os 30 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** <descrever sensibilidade do modelo: pequeno (< 1.000), médio (1.000 a 10.000), grande (> 10.000)>
- **Número de classes:** <binário vs. multiclasse; degradação esperada conforme aumenta o número de classes>
- **Proporção entre features categóricas e numéricas:** <baixa vs. alta; impacto na codificação e no tempo de treino>
- **Presença de valores ausentes:** <com NaN vs. sem NaN; estratégia de imputação adotada>

## 4. Métricas alcançadas

Tabela agregada nos 30 datasets do TabArena. Reportar média, desvio padrão e intervalo de confiança de 95% via bootstrap (1.000 reamostragens).

| Métrica | Média | Desvio | IC 95% (bootstrap) | Ranking médio |
|---|---|---|---|---|
| AUC OvO | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| Accuracy | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| G-Mean | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| Cross-Entropy | <0,0000> | <0,0000> | <[0,0000; 0,0000]> | <0,0> |
| Tempo total (s) | <0,0> | <0,0> | <[0,0; 0,0]> | <0,0> |

### Resultados por regime

- **Tamanho:** pequeno: AUC=<...>; médio: AUC=<...>; grande: AUC=<...>
- **Número de classes:** binário: AUC=<...>; multiclasse: AUC=<...>
- **Proporção categórica:** baixa: AUC=<...>; alta: AUC=<...>
- **Missing values:** com NaN: AUC=<...>; sem NaN: AUC=<...>

## 5. Dados de avaliação

- **Origem:** 30 datasets do TabArena-v0.1 (NeurIPS 2025), via OpenML.
- **Distribuição por regime:** 10 pequenos + 10 médios + 10 grandes.
- **Estratégia de split:** 70/30 estratificado por classe, seed=<n>.
- **Pré-processamento aplicado:** <descrever imputação, codificação categórica e escalonamento>.
- **Lista dos datasets utilizados:** <preencher com nome, OpenML task ID, n, n_features, n_classes, regime; ver tabela do relatório>.

## 6. Dados de treino e pré-treino

- **Modelo é foundation model pré-treinado, treinado do zero ou híbrido?** TabPFN-2.5 trata-se essencialmente de um Foundation Model tabular pré-treinado maciçamente.
- **Origem dos dados de pré-treino (se aplicável):** Os conhecimentos prévios e distribuições base utilizadas na fase massiva de metatransformação do modelo provêm integralmente de milhões de distribuições sintéticas que emulam dados tabulares heterogêneos
- **Origem dos dados de treino direto (se aplicável):** Existe também a variante do modelo chamada Real-TabPFN-2.5 que inclui aperfeiçoamento residual e fine-tuning contínuo derivado de 43 conjuntos puros originários do mundo real curados no ecossistema OpenML e Kaggle.
- **Possíveis vieses herdados do pré-treino:** <descrever; relevante para foundation models>

## 7. Análise quantitativa

- **Posição no ranking médio entre os 15 sistemas avaliados** (10 modelos atribuíveis + 3 baselines + 2 AutoGluon): <x de 15>
- **Friedman + Nemenyi:** <descrever resultado global e os grupos estatisticamente equivalentes; citar o diagrama de diferença crítica>
- **Bayesian signed-rank com ROPE = 0,01 em AUC:** <descrever pares com p_equivalente acima de 0,95; pares onde o modelo do grupo é claramente melhor ou pior>
- **Comparação com AutoGluon:** <delta de AUC e custo computacional vs. preset default e vs. preset extreme 4h>
- **Quebra por regime:** <em quais regimes o modelo do grupo vence; em quais perde; provável explicação alinhada à arquitetura>

## 8. Considerações éticas

- **Riscos de uso indevido:** <ex.: viés herdado dos dados de pré-treinamento sintético, decisões opacas em domínios sensíveis>
- **Fairness por classe:** <recall e precisão por classe; classes minoritárias com baixo recall>
- **Dependência de licença de pesos pré-treinados:** <ex.: TabPFN-2.5 tem licença não-comercial; uso em produção exige avaliação jurídica>
- **Impacto ambiental:** <energia consumida durante tuning; latência de inferência; trade-off entre qualidade e custo>
- **Recomendações de auditoria:** <ex.: comparar predições com baseline interpretável como EBM antes de deploy>

## 9. Avisos e recomendações

- **Quando usar este modelo:** <regimes em que mostrou melhor desempenho ou melhor custo-benefício>
- **Quando NÃO usar este modelo:** <regimes onde os baselines vencem ou onde restrições operacionais inviabilizam o uso>
- **Alternativas recomendadas em cada caso:** <ex.: para n acima de 50K, usar LightGBM TD; para datasets com cardinalidade categórica alta, usar CatBoost TD; para AutoML genérico, usar AutoGluon default>

## 10. Reprodutibilidade

- **Ambiente:** Python <3.11>, dependências fixadas em `pyproject.toml`.
- **Hardware utilizado:** GPU NVIDIA GeForce RTX 5060 ti 16GB, 32GB de RAM, tempo total de execução>
- **Comandos para reproduzir:**
  ```bash
  uv sync
  python -m src.pipeline.run_all --include-group-model --seed 42
  ```
- **Hash do commit:** <git rev-parse HEAD>

## 11. Referências

- L. Grinsztajn, K. Flöge, O. Key, F. Birkel, P. Jund, B. Roof, B. Jäger, D. Safaric,S. Alessi, A. Hayler, M. Manium, R. Yu, F. Jablonski, S. B. Hoo, A. Garg,J. Robertson, M. Bühler, V. Moroshan, L. Purucker, C. Cornu, L. C. Wehrhahn,A. Bonetto, B. Schölkopf, S. Gambhir, N. Hollmann, and F. Hutter, “Tabpfn-2.5:Advancing the state of the art in tabular foundation models,” 2026. [Online].Available: https://arxiv.org/abs/2511.08667
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.
- Demsar, J. (2006). Statistical comparisons of classifiers over multiple datasets. JMLR.
- Benavoli, A., Corani, G., Demsar, J., Zaffalon, M. (2017). Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. JMLR.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
