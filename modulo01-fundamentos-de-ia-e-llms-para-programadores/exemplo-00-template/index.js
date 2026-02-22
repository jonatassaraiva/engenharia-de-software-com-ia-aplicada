import tf, { train } from '@tensorflow/tfjs-node';

// =============================================================================
// ARQUITETURA DA REDE NEURAL
// =============================================================================
// Uma rede neural é composta por camadas de neurônios interligados que aprendem
// a mapear entradas (características de pessoas) para saídas (categorias de plano).
//
// Fluxo dos dados:
//   [Entrada: 7 características] → [Camada Oculta: 80 neurônios] → [Saída: 3 categorias]
//
// Camadas desta rede:
//   1. Camada de Entrada  → recebe 7 características por pessoa
//   2. Camada Oculta      → 80 neurônios aprendem padrões intermediários
//   3. Camada de Saída    → 3 neurônios geram probabilidades para cada categoria
//                           (premium | medium | basic)
// =============================================================================
async function trainModel(inputShape, outputShape) {
    // 1 - Criamos um modelo sequencial, que é uma pilha linear de camadas.
    const model = tf.sequential();

    // 2 - Camada oculta
    // Adicionamos uma camada densa (fully connected) com 80 neurônios e função de ativação ReLU.
    // - units: 80 → número de neurônios nesta camada, responsáveis por aprender padrões nos dados.
    // - activation: 'relu' → função que elimina valores negativos (max(0, x)), ajudando o modelo
    //   a aprender relações não-lineares sem o problema de vanishing gradient.
    // - inputShape: [7] → formato de entrada com 7 características por pessoa:
    //     [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
    //   onde a idade é normalizada entre 0 e 1, e as demais são valores one-hot encoded.
    model.add(tf.layers.dense({
        units: 80,
        activation: 'relu',
        inputShape: [7]
    }));

    // 3 -
    // Adicionamos uma camada de saída densa com 3 neurônios (correspondente às 3 categorias: premium, medium, basic).
    // - units: 3 → número de neurônios na camada de saída, correspondendo às 3 categorias que queremos prever.
    // - activation: 'softmax' → função que converte as saídas em probabilidades, garantindo que a soma das saídas seja 1.
    //   Isso é essencial para problemas de classificação multi-classe, onde cada neurônio representa a probabilidade de uma classe específica.
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
     }));

    // Para compilar o modelo, vamos usar o otimizador Adam, a função de perda categoricalCrossentropy
    // (pois é um problema de classificação multi-classe) e a métrica de acurácia para avaliar o desempenho
    // durante o treinamento.
    // O otimizador Adam é um algoritmo que ajusta os pesos da rede neural durante o treino para minimizar erros e ajusta a taxa de aprendizado individualmente para cada peso.
    // A função de perda categoricalCrossentropy é usada para medir a diferença entre as distribuições de probabilidade previstas pelo modelo e as distribuições reais (labels) em problemas de classificação multi-classe.
    // A métrica de acurácia é usada para avaliar a proporção de previsões corretas feitas pelo modelo em relação ao total de previsões.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamos o modelo usando os dados de entrada (inputXs) e as labels (outputYs).
    // - epochs: 100 → número de vezes que o modelo passará por todo o conjunto de dados durante o treinamento.
    //   Um número maior de epochs pode levar a um melhor aprendizado, mas também pode causar overfitting se for muito alto.
    await model.fit(inputXs, outputYs, {
        epochs: 100, // Número de vezes que o modelo passará por todo o conjunto de dados durante o treinamento.
        shuffle: true, // Embaralha os dados a cada época para melhorar a generalização do modelo.
        verbose: 0, // Define o nível de verbosidade do treinamento (0 = sem logs, 1 = barra de progresso, 2 = uma linha por época).
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if ((epoch + 1) % 10 === 0) { // A cada 10 épocas, imprime o progresso
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${(logs.acc * 100).toFixed(2)}%`);
                }
            }
        }
    });

    return model;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();

// Quantos mais dados de treino tivermos, melhor o modelo pode aprender a generalizar e fazer previsões precisas.
const model = trainModel(inputXs.shape[1], outputYs.shape[1]);
