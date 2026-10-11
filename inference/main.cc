#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <span>
#include <stdexcept>

using namespace std;
using tensor = vector<float>;
const float EPSILON = 1e-5f;

tensor layerNorm(
    std::span<const float> ogEmbeddings, // span just give you acces to see the vectors without copying it
    std::span<const float> weights,
    std::span<const float> biases)
{
    assert(ogEmbeddings.size() == weights.size());
    assert(ogEmbeddings.size() == biases.size());

    int n = static_cast<int>(ogEmbeddings.size());

    // Step 1: Calculate mean
    float mean = 0.0f;

    for (float v : ogEmbeddings)
    {
        mean += v;
    }

    mean /= static_cast<float>(n);

    // step 2: calculate variance
    float variance = 0.0f;

    for (float v : ogEmbeddings)
    {
        float d = v - mean;
        variance += d * d;
    }
    variance /= static_cast<float>(n);

    float invStd = 1.0f / std::sqrt(variance + EPSILON);

    tensor output(n);

    for (int i = 0; i < n; i++)
    {
        float normalized = (ogEmbeddings[i] - mean) * invStd;

        output[i] = normalized * weights[i] + biases[i];
    }

    return output;
}

struct QKVWeights
{
    tensor weights;
    tensor bias;
};

tensor linearProjection(
    std::span<const float> input,
    const QKVWeights &qkv)
{

    const int inputSize = 768;
    const int outputSize = 2304;

    assert(input.size() == inputSize);
    assert(qkv.weights.size() == inputSize * outputSize);
    assert(qkv.bias.size() == outputSize);

    tensor output(outputSize, 0.0f);

    for (int j = 0; j < outputSize; j++)
    {
        float sum = qkv.bias[j];

        for (int i = 0; i < inputSize; i++)
        {
            sum += input[i] *
                   qkv.weights[i * outputSize + j];
        }

        output[j] = sum;
    }

    return output;
}

QKVWeights
loadQKVWeights()
{
    QKVWeights qkv;

    ifstream weightFile(
        "../weights/transformer.h.0.attn.c_attn.weight.txt");

    ifstream biasFile(
        "../weights/transformer.h.0.attn.c_attn.bias.txt");

    if (!weightFile.is_open() || !biasFile.is_open())
    {
        throw runtime_error("Error opening QKV weight or bias file");
    }

    float value;

    while (weightFile >> value)
    {
        qkv.weights.push_back(value);
    }

    while (biasFile >> value)
    {
        qkv.bias.push_back(value);
    }

    return qkv;
}

float dotProduct(
    std::span<const float> Q,
    std::span<const float> K)
{
    assert(Q.size() == K.size());

    float sum = 0.0f;

    for (int i = 0; i < Q.size(); i++)
    {
        sum += Q[i] * K[i];
    }

    return sum;
}

tensor softmax(std::span<const float> scores)
{
    tensor output(scores.size());
    float sum = 0.0f;

    for (int i = 0; i < scores.size(); i++)
    {
        output[i] = std::exp(scores[i]);
        sum += output[i];
    }

    for (int i = 0; i < output.size(); i++)
    {
        output[i] /= sum;
    }

    return output;
}

tensor weightedSum(
    std::span<const float> weights,
    const std::vector<tensor> &values)
{
    assert(weights.size() == values.size());

    const int headDim = 64;
    tensor output(headDim, 0.0f);

    for (int token = 0; token < weights.size(); token++)
    {
        for (int i = 0; i < headDim; i++)
        {
            output[i] += weights[token] * values[token][i];
        }
    }

    return output;
}

tensor calculateAttention(
    int queryIndex,
    int headIndex,
    const std::vector<std::vector<tensor>> &allHeadQ,
    const std::vector<std::vector<tensor>> &allHeadK,
    const std::vector<std::vector<tensor>> &allHeadV)
{
    std::vector<float> scores;

    for (int token = 0; token <= queryIndex; token++)
    {
        float score = dotProduct(
            allHeadQ[queryIndex][headIndex],
            allHeadK[token][headIndex]);

        score /= std::sqrt(64.0f);
        scores.push_back(score);
    }

    tensor weights = softmax(scores);

    std::vector<tensor> values;

    for (int token = 0; token <= queryIndex; token++)
    {
        values.push_back(allHeadV[token][headIndex]);
    }

    return weightedSum(weights, values);
}



tensor concatenateHeads(const std::vector<tensor>& headOutputs)
{
    tensor combined;

    for (const tensor& head : headOutputs)
    {
        combined.insert(
            combined.end(),
            head.begin(),
            head.end()
        );
    }

    return combined;
}


int main()
{

    string path = "../weights/transformer.wte.weight.txt";
    string path2 = "../weights/transformer.wpe.weight.txt";
    string lnWeightPath = "../weights/transformer.h.0.ln_1.weight.txt";
    string lnBiasPath = "../weights/transformer.h.0.ln_1.bias.txt";

    ifstream file(path);
    ifstream file2(path2);
    ifstream lnWeightFile(lnWeightPath);
    ifstream lnBiasFile(lnBiasPath);

    if (!file.is_open() || !file2.is_open() ||
        !lnWeightFile.is_open() || !lnBiasFile.is_open())
    {
        cout << "Failed to open weights file!" << endl;
        return 1;
    }
    vector<float> embeddings;
    vector<float> positionalEmbeddings;
    vector<float> gamma;
    vector<float> beta;

    float value;
    float positionalvalues;
    float lnValue;

    while (file >> value)
    {
        embeddings.push_back(value);
    }

    while (file2 >> positionalvalues)
    {

        positionalEmbeddings.push_back(positionalvalues);
    }

    while (lnWeightFile >> lnValue)
    {
        gamma.push_back(lnValue);
    }

    while (lnBiasFile >> lnValue)
    {
        beta.push_back(lnValue);
    }

    file.close();
    file2.close();
    lnWeightFile.close();
    lnBiasFile.close();

    cout << "Gamma values: " << gamma.size() << endl;
    cout << "Beta values: " << beta.size() << endl;

    // int tokenId = 3797;
    // int postion = 1;

    // int tokenStart = tokenId * 768;
    // int positionStart = postion * 768;

    // for (int i = 0; i < 768; i++)
    // {
    //     finalEmbedding[i] = embeddings[tokenStart + i] + positionalEmbeddings[positionStart + i];
    // }

    vector<int> tokenIds = {32, 3797, 318};

    vector<vector<float>> inputEmbeddings;
    vector<vector<float>> normalizedEmbeddings;

    for (int tokenIndex = 0; tokenIndex < tokenIds.size(); tokenIndex++)
    {
        int tokenId = tokenIds[tokenIndex];

        int tokenStart = tokenId * 768;
        int positionStart = tokenIndex * 768;

        vector<float> finalEmbedding(768);

        for (int i = 0; i < 768; i++)
        {
            finalEmbedding[i] =
                embeddings[tokenStart + i] +
                positionalEmbeddings[positionStart + i];
        }

        // Push each token's embedding before finalEmbedding goes out of scope
        inputEmbeddings.push_back(finalEmbedding);
    }

    // Loop ke baad LayerNorm call karo

    for (int i = 0; i < inputEmbeddings.size(); i++)
    {

        vector<float> normalized = layerNorm(
            inputEmbeddings[i],
            gamma,
            beta);

        normalizedEmbeddings.push_back(normalized);
    }

    const int HEAD_DIM = 64;

    vector<vector<tensor>> allHeadQ;
    vector<vector<tensor>> allHeadK;
    vector<vector<tensor>> allHeadV;

    QKVWeights qkv = loadQKVWeights();

    const int EMBED_SIZE = 768;

    for (int tokenIndex = 0;
         tokenIndex < normalizedEmbeddings.size();
         tokenIndex++)
    {
        const vector<float> &normalizedToken =
            normalizedEmbeddings[tokenIndex];

        tensor qkvOutput = linearProjection(normalizedToken, qkv);

        // Abhi yahin se Q, K, V split karenge

        span<const float> Q(qkvOutput.data(), 768);

        span<const float> K(qkvOutput.data() + 768, 768);

        span<const float> V(qkvOutput.data() + 1536, 768);

        const int HEAD_DIM = 64;

        // First head: Q, K, V ke pehle 64 values

        vector<tensor> tokenHeadsQ;
        vector<tensor> tokenHeadsK;
        vector<tensor> tokenHeadsV;

        for (int head = 0; head < 12; head++)
        {
            int start = head * HEAD_DIM;

            span<const float> headQ(Q.data() + start, HEAD_DIM);
            span<const float> headK(K.data() + start, HEAD_DIM);
            span<const float> headV(V.data() + start, HEAD_DIM);

            tokenHeadsQ.emplace_back(headQ.begin(), headQ.end());
            tokenHeadsK.emplace_back(headK.begin(), headK.end());
            tokenHeadsV.emplace_back(headV.begin(), headV.end());
        }

        allHeadQ.push_back(tokenHeadsQ);
        allHeadK.push_back(tokenHeadsK);
        allHeadV.push_back(tokenHeadsV);
    }

vector<tensor> catHeadOutputs;

for (int head = 0; head < 12; head++)
{
    tensor output = calculateAttention(
        1,
        head,
        allHeadQ,
        allHeadK,
        allHeadV
    );

    catHeadOutputs.push_back(output);
}

cout << "Number of head outputs: "
     << catHeadOutputs.size() << endl;

     tensor combinedOutput = concatenateHeads(catHeadOutputs);

cout << "Combined output size: "
     << combinedOutput.size() << endl;


    return 0;
}