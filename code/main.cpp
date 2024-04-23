#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "npp/npp.h"

using namespace std;

#define DATASET_FILENAME "winequality_white_normalized.txt"

#define PERCENT_TRAINING 0.8
#define PERCENT_TESTING 0.2
#define INPUTS 11
#define OUTPUTS 11
#define HIDDEN 18
#define LAYERS 4

#define MAX_SEEDS 10
#define MAX_EPOCHS 10000

void createNet() {
    // TODO: add this function to create the networks
}

/**
 *
 * @param filename
 * @param dataset
 * @return
 */
bool readDataset(const string &filename, vector<string> *dataset) {
    ifstream file(filename);
    if (!file || !file.is_open()) {
        return false;
    }
    std::string line;
    while (getline(file, line)) {
        if (line.length() > 5) {
            dataset->push_back(line);
        }
    }
    shuffle(dataset->begin(), dataset->end(), default_random_engine(time(nullptr)));
    return true;
}

/**
 *
 * @param dataset
 * @param seed
 * @param numSeeds
 * @param trainingData
 * @param testingData
 */
void splitDataset(const vector<string> dataset, const int seed, const int numSeeds, vector<string> *trainingData,
                  vector<string> *testingData) {
    const int datasetCount = dataset.size() / numSeeds;
    const int trainingOffset = (seed - 1) * datasetCount;
    const int trainingDataCount = datasetCount * PERCENT_TRAINING;
    const int testingOffset = trainingOffset + trainingDataCount;
    vector<string> trainingVector(&dataset[trainingOffset], &dataset[testingOffset - 1]);
    for (auto data: trainingVector) {
        trainingData->emplace_back(data);
    }
    vector<string> testingVector(&dataset[testingOffset], &dataset[trainingOffset + datasetCount - 1]);
    for (auto data: testingVector) {
        testingData->emplace_back(data);
    }

}

/**
 *
 * @param data
 * @param inVec
 * @param outVec
 * @param tarVec
 */
static inline void initializeVectors(const string &data, float *inVec, float *outVec,
                                     float *tarVec) {
    istringstream ss(data);
    string token;
    int i = 0;
    while (getline(ss, token, ';') && i < INPUTS) {
        inVec[i] = stof(token);
        i++;
    }
    fill_n(outVec, OUTPUTS, 0);
    tarVec[stoi(token)] = 1;
}

/**
 *
 * @param seed
 * @param trainingData
 * @param testingData
 */
void runSeed(const int seed, vector<string> trainingData,
             vector<string> testingData) {
    const string filename = "startnet_seed_" + to_string(seed) + ".net";
    cout << filename << endl;
    const auto net = new Net();
    net->load_net(filename.c_str());
    const auto inVec = new float[net->topo_data.in_count];
    const auto outVec = new float[net->topo_data.out_count];
    const auto tarVec = new float[net->topo_data.out_count];
    float uparams[10];

    net->save_net("start_uparam1_" + to_string(uparams[0]) + "_uparam2_" +
                  to_string(uparams[1]) + "_uparam3_" + to_string(uparams[2]) +
                  "_seed_" + to_string(seed) + ".net");

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        float tss = 0.0;
        for (const auto data: trainingData) {
            initializeVectors(data, inVec, outVec, tarVec);
            net->forward_pass(inVec, outVec);
            for (int i = 0; i < net->topo_data.out_count; i++) {
                const double error = outVec[i] = outVec[i] - tarVec[i];
                tss += error * error;
            }
            net->backward_pass(outVec, inVec);
        }
        //
        float tssTesting = 0, tssCorrect;
        int correct = 0;
        //
        for (const auto data: testingData) {
            tssCorrect = 0;
            initializeVectors(data, inVec, outVec, tarVec);
            net->forward_pass(inVec, outVec);
            for (int i = 0; i < net->topo_data.out_count; i++) {
                float error = outVec[i] - tarVec[i]; // dE_total/dout_oi
                tssCorrect += error * error; // E_total
                tssTesting += tssCorrect;
            }
            if (tssCorrect < 0.15)
                correct++;
        }
        cout << seed << "/" << epoch << " correct=" << correct << endl;
        net->update_weights();
    }

    delete[] inVec;
    delete[] outVec;
    delete[] tarVec;
}

int main() {
    vector<string> dataset;
    if (!readDataset(DATASET_FILENAME, &dataset)) {
        cerr << "Could not read dataset " << DATASET_FILENAME << endl;
        return 1;
    }
    vector<thread> threads;
    for (int seed = 1; seed <= MAX_SEEDS; seed++) {
        // Read the dataset in from the file
        vector<string> trainingData, testingData;
        splitDataset(dataset, seed, MAX_SEEDS, &trainingData, &testingData);
        //runSeed(seed, trainingData, testingData);
        threads.push_back(thread(runSeed, seed, trainingData, testingData));
    }
    for (auto &thread: threads) {
        thread.join();
    }
    return 0;
}
