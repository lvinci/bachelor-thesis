#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <thread>

#include "npp/npp.h"

using namespace std;

#define DATASET_FILENAME "magic_normalized.txt"
#define SEEDS 10
#define EPOCHS 10000

#define PERCENT_TRAINING 0.8
#define INPUTS 10
#define OUTPUTS 2

/**
 * Represents a row in a dataset.
 * Every row has a number of floats which matches the INPUTS of the neural network.
 * Furthermore the classification is stored as a char.
 */
typedef struct {
    float input[INPUTS];
    char classification;
} DatasetRow;

/**
 * Attempts to parse a DatasetRow from the given string.
 * @param line string to be parsed
 * @param row row that will be used to store the parsed dataset row
 * @return true if the string could be parsed, false otherwise
 */
bool parseDatasetRow(const string &line, DatasetRow *row) {
    // Create a string stream to parse the line
    stringstream ss(line);
    string substr;
    // Parse the first floats separated by ';'
    for (int i = 0; i < INPUTS; ++i) {
        if (!getline(ss, substr, ';') || !(istringstream(substr) >> row->input[i])) {
            return false; // If conversion fails or delimiter is missing, return false
        }
    }
    // Parse the last character (classification)
    if (!getline(ss, substr, ';') || substr.size() != 1) {
        return false;
    }
    row->classification = substr[0];
    return true;
}

/**
 * Reads the dataset from the file with the given filename
 * @param filename name (relative path) of the file to be read
 * @param dataset vector to save the parsed dataset rows
 * @return true if the dataset could be read, false otherwise
 */
bool readDataset(string filename, vector<DatasetRow> *dataset) {
    ifstream file(filename);
    if (!file || !file.is_open()) {
        return false;
    }
    string line;
    while (getline(file, line)) {
        DatasetRow row;
        if (parseDatasetRow(line, &row)) dataset->push_back(row);
    }
    shuffle(dataset->begin(), dataset->end(), default_random_engine(time(nullptr)));
    return true;
}

/**
 *
 * @param dataset
 * @param numPartitions
 * @param partition
 * @param trainingset
 * @param testingset
 */
void
partitionDataset(const vector<DatasetRow> &dataset, const int numPartitions, const int partition,
                 vector<DatasetRow> *trainingset,
                 vector<DatasetRow> *testingset) {
    // Calculate the sizes of the training and testing sets
    const size_t partitionSize = dataset.size() / numPartitions;
    const size_t trainingsetSize = partitionSize * PERCENT_TRAINING;
    const size_t trainingsetStart = partitionSize * (partition - 1);
    const size_t testingsetStart = trainingsetStart + trainingsetSize + 1;
    // Copy the training set rows into the trainingset
    auto trainingBegin = dataset.begin() + trainingsetStart;
    auto trainingEnd = dataset.begin() + testingsetStart - 1;
    trainingset->insert(trainingset->end(), trainingBegin, trainingEnd);
    // Copy the testing set rows into the testingset
    auto testingBegin = dataset.begin() + testingsetStart;
    auto testingEnd = dataset.begin() + trainingsetStart + partitionSize - 1;
    testingset->insert(testingset->end(), testingBegin, testingEnd);
}

/**
 *
 * @param row
 * @param inVec
 * @param outVec
 * @param tarVec
 */
static inline void initializeVectors(const DatasetRow &row, float *inVec, float *outVec,
                                     float *tarVec) {
    copy(row.input, row.input + INPUTS, inVec);
    fill_n(outVec, OUTPUTS, 0);
    tarVec[0] = row.classification == 'g';
    tarVec[1] = row.classification == 'h';
}

static void runSeed(const int seed, const vector<DatasetRow> &trainingset, const vector<DatasetRow> &testingset) {
    const string filename = "startnet_seed_" + to_string(seed) + ".net";
    const auto net = new Net();
    net->load_net(filename.c_str());
    const auto inVec = new float[net->topo_data.in_count];
    const auto outVec = new float[net->topo_data.out_count];
    const auto tarVec = new float[net->topo_data.out_count];
    float uparams[10];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float tss = 0.0;
        for (const auto data: trainingset) {
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
        for (const auto data: testingset) {
            tssCorrect = 0;
            initializeVectors(data, inVec, outVec, tarVec);
            net->forward_pass(inVec, outVec);
            for (int i = 0; i < net->topo_data.out_count; i++) {
                float error = outVec[i] - tarVec[i]; // dE_total/dout_oi
                tssCorrect += error * error; // E_total
                tssTesting += tssCorrect;
            }
            if (tssCorrect < 0.05)
                correct++;
        }
        net->update_weights();
        printf("%d/%d correct=%d\n", seed, epoch, correct);
    }
    delete[] inVec;
    delete[] outVec;
    delete[] tarVec;
}

int main() {
    // Read in dataset from the dataset textfile
    vector<DatasetRow> dataset;
    if (!readDataset(DATASET_FILENAME, &dataset)) {
        printf("Could not read dataset from file %s. Does the file exist?\n", DATASET_FILENAME);
        return 1;
    }
    printf("Read dataset from file %s with %zu rows.\n", DATASET_FILENAME, dataset.size());
    //
    vector<thread> threads;
    for (int seed = 1; seed <= SEEDS; seed++) {
        // Partition the dataset into a training set and a testing set for this seed
        vector<DatasetRow> trainingset, testingset;
        partitionDataset(dataset, SEEDS, seed, &trainingset, &testingset);
        printf("starting thread for seed %d with %zu training and %zu testing rows\n", seed, trainingset.size(),
               testingset.size());
        //
        runSeed(seed, trainingset, testingset);
        //threads.push_back(thread(runSeed, seed, trainingset, testingset));
    }
    /* for (auto &thread: threads) {
         thread.join();
     }*/
    return 0;
}