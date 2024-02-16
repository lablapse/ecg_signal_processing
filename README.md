# Detecção e classificação de patologias cardíacas em eletrocardiograma utilizando redes neurais profundas


Doenças cardiovasculares vêm figurando, no decorrer das últimas décadas, dentre as principais causas de morte registradas no mundo. Tais doenças podem ser detectadas/diagnosticadas por um profissional de saúde através de um exame não invasivo denominado eletrocardiograma (ECG), o qual fornece uma representação da atividade elétrica do coração. Entretanto, a interpretação do ECG não é uma tarefa trivial, demandando considerável experiência do profissional de saúde na identificação apropriada de alterações morfológicas. Nesse contexto, técnicas de aprendizado de máquina (em especial, aprendizado profundo) vêm sendo amplamente utilizadas, na área de eletrofisiologia clínica, para criar modelos matemáticos capazes de detectar automaticamente padrões que caracterizam a presença ou ausência de patologias cardíacas. Contudo, apesar do bom desempenho alcançado, poucos trabalhos apresentados até então utilizam conjuntos de dados públicos no treinamento das arquiteturas, dificultando assim a validação dos resultados, comparações de desempenho e proposição de melhorias. Diante disso, o presente trabalho foca sobre o desenvolvimento de uma ferramenta computacional para classificação automática multirrótulos de 5 superclasses de patologias cardíacas. Especificamente, as arquiteturas introduzidas em (RAJPURKAR et al., 2017) e (RIBEIRO et al., 2020) são implementadas, utilizando linguagem Python juntamente as bibliotecas Tensorflow e Keras, e treinadas considerando um conjunto de dados público já rotulado por cardiologistas, denominado PTB-XL (WAGNER et al., 2020). O desempenho dos modelos obtidos (após o treinamento) é avaliado através da matriz de confusão multirrótulos (MLCM), a partir da qual as métricas precisão, sensibilidade e F1-score assim como seus valores médios micro, macro e weighted podem ser computados, conforme metodologia introduzida em (HEYDARIAN; DOYLE; SAMAVI, 2022). Ainda, comparações envolvendo o custo computacional para a implementação e treinamento das arquiteturas são apresentadas, abordando a quantidade de parâmetros, o número de épocas e o tempo transcorrido até o encerramento do treinamento. Os resultados obtidos mostraram que o modelo de (RIBEIRO et al., 2020) apresenta um desempenho similar, ou ainda ligeiramente melhor, para as diferentes métricas consideradas do que aquele obtido a partir da arquitetura de (RAJPURKAR et al., 2017); sobretudo, ao se levar em conta o custo computacional envolvido.

# Contexto geral


Este trabalho reimplementa, em [Pytorch](https://pytorch.org), o código desenvolvido em um [trabalho realizado](https://github.com/lablapse/ecg_signal_processing.git) anteriormente. Desta forma, os códigos traduzidos se disponibilizam nos seguintes *scripts*:

- As arquiteturas de classificação de eletrocardiograma (ECG) estão disponibilizadas em [utils_torch.py](utils_torch.py) e em [utils_lightning.py](utils_lightning.py). O primeiro *script* contém as informações para construir os modelos. O segundo *script* utiliza uma biblioteca denominada [pytorch lightning](utils_lightning.py) para realmente criar os modelos, inserindo, assim, a função custo, *learning rate* e outras informações, além de facilitar questões envolvendo treinamento e manipulação das redes neurais.

- A busca exaustiva de hiperparâmetros pode ser acessada em [grid_search_torch.py](grid_search_torch.py). Durante a modificação do _script_, algumas coisas não foram adaptadas - à exemplo disso, algumas informações não são coletadas e salvas no arquivo _grid_search_results_torch.csv_ conforme feito no trabalho implementado anteriormente.


# Principais problemas

Como a biblioteca com a qual o trabalho foi originalmente desenvolvido - [keras](https://www.tensorflow.org/guide/keras?hl=pt-br) - e a biblioteca aonde o novo código foi escrito são diferentes, os resultados obtidos pelas pesquisas de hiperparâmetros se mostraram consideravelmente distintos. De uma maneira geral, o trabalho em [Pytorch](https://pytorch.org) performou pior. Para encontrar aonde os _frameworks_ diferiam em performance, diversos estudos foram realizados, porém, ainda não se obtiveram resultados realmente comparáveis entre as duas aplicações.

## Conceitos e comportamentos de classes e funções

Algumas operações diferem conceitualmente entre as bibliotecas. Por exemplo, o inicializador, que, neste caso, foi utilizado o qual geralmente se denomina '[He normal](https://arxiv.org/abs/1502.01852)'. Para o _Keras_, [este inicializador](https://keras.io/api/layers/initializers/) utiliza como base uma distribuição normal truncada, enquanto, para o _Pytorch_, a distribuição base não é truncada. Esta diferença por si só já é o suficiente para alterar completamente a performance das duas redes neurais por alterar os valores de inicialização. Além disso, lendo as documentações, sugerem-se diferenças de comportamento entre a operação de _dropout_, que vale ser investigado. 

## Análise de _forward_, _backward_ e os gradientes

### _Forward_

Para garantir que as operações utilizadas resultam em valores equiparáveis durante o processo de _forward_, foi realizado o seguinte procedimento:

- Foi gerado e salvo um array aleatório de tipo _float32_, utilizando-se a função [_rand_](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html), com dimensões (amostras, canais, tamanho), para ser utilizado na análise a seguir. 
- Criou-se três ambientes [conda](https://www.anaconda.com/download), um contendo o _keras_, outro contendo o _pytorch_ e outro sem nenhum desses _frameworks_.
- Em dois _scripts_ separados, um para cada ambiente _conda_ que contém _framework_, foi criado uma função com diversas operações, entre elas convolução unidimensional e _batch normalization_ que, ao receber o _array_ gerado anteriormente, o inseriria nas operações e salvaria os resultados individuais.
- Um terceiro _script_, feito sem nenhum dos dois _frameworks_, carrega os resultados gerados anteriormente e os compara, utilizando as [normas](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) dos novos vetores, e salva estes resultados em uma pasta.

Foi constatado que as operações analisadas geram resultados semelhantes e as informações podem ser acessadas na pasta [comparing_forward_keras_torch](comparing_forward_keras_torch).

### _Backward_ e gradientes

A análise de _backward_ e de gradiente é mais complexa que a anterior, afinal, para ela funcionar corretamente deve-se possuir uma função custo e um otimizador, além de ser necessário que seja realizada uma operação de _forward_ poder fazer a etapa de _backward_. Como este passo é mais intrincado que o anterior, não será tão detalhado, porém, a lógica de ambas as análises opera de maneira semelhante: um _script_ para cada ambiente e um teiceiro _script_ sem nenhum dos dois _frameworks_. Neste caso foi montado um modelo com diversas operações e os valores salvos por cada _script_ correspondem à uma opearação de _forward_ nestes modelos



Ademais, a operação de _batch normalization_ do _keras_ apresenta comportamentos distintos durante o treinamento e a validação, o que serve de obstáculo para a análise direta 

O que dizer -> Falar que este trabalho se deriva de outro trabalho.
               Falar que o trabalho original implementa duas arquiteturas para a classificação de ECG.
               Falar que o trabalho original implementa um busca exaustiva de hiperparâmetros.
               Falar que este trabalho implementa praticametne tudo o que o está disponível no trabalho original, porém, ao invés de torch e keras.
               Falar que como a conversão do keras para o torch nao é direta, fragmentos de funções e classes foram comparadas.
               Falar sobre a conclusão de algumas diferenças (he normal, talvez dropout)
               Alertar o próximo programador sobre coisas como a dimensão dos filtros em torch e keras.
               Alertar o próximo programador sobre como em duas ocasiões distintas o ambiente do tensorflow simplesmente recusou o comando de run do script.
               Alertar o próximo programador sobre não instalar um ambiente com o torch e o keras ao mesmo tempo.
