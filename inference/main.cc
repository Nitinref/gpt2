#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <span>

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
        cout << "Normalized values: " << normalized.size() << endl;
    }
    cout << "Normalized tokens stored: "
         << normalizedEmbeddings.size() << endl;

    cout << "Values in first normalized token: "
         << normalizedEmbeddings[0].size() << endl;
    cout << "Total tokens processed: " << inputEmbeddings.size() << endl;
    cout << "Numbers per token: " << inputEmbeddings[0].size() << endl;

    return 0;
}