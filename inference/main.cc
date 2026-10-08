#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main()
{

    string path = "../weights/transformer.wte.weight.txt";

    ifstream file(path);

    if (!file.is_open())
    {
        cout << "Failed to open weights file!" << endl;
        return 1;
    }

    vector<float> embeddings;

    float value;

    while (file >> value)
    {
        embeddings.push_back(value);
    }

    file.close();

    cout << "Loaded values: " << embeddings.size() << endl;
   vector<int> tokenIds = {32, 3797, 318, 319, 262, 2603, 13};

for (int tokenId : tokenIds) {
    int start = tokenId * 768;

    cout << "\nToken ID: " << tokenId << endl;
    cout << "Start index: " << start << endl;

    cout << "First 5 embedding values:" << endl;

    for (int i = 0; i < 5; i++)
    {
        cout << embeddings[start + i] << endl;
    }
}
    return 0;
}