#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>

#include "npp/npp.h"
#include "threadpool/threadpool.h"

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
bool readDataset(string filename, vector<DatasetRow> *dataset, const int randomSeed) {
    ifstream file(filename);
    if (!file || !file.is_open()) {
        return false;
    }
    string line;
    while (getline(file, line)) {
        DatasetRow row;
        if (parseDatasetRow(line, &row)) dataset->push_back(row);
    }
    shuffle(dataset->begin(), dataset->end(), default_random_engine(randomSeed));
    return true;
}

/**
 * Partitions the given dataset into training and testing set.
 * Additionally, only a part of the given dataset can be used.
 * @param dataset complete dataset to be partitioned
 * @param numPartitions number of partitions (1 if you want the full dataset)
 * @param partition number of the current partition
 * @param trainingset training set vector to be initialized
 * @param testingset testing set vector to be initialized
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
 * Initializes the vectors for a neural network pass using the given dataset
 * @param row dataset row to be used
 * @param inVec vector for the inputs
 * @param outVec vector for the outputs
 * @param tarVec target vector
 */
static inline void initializeVectors(const DatasetRow &row, float *inVec, float *outVec,
                                     float *tarVec) {
    copy(row.input, row.input + INPUTS, inVec);
    fill_n(outVec, OUTPUTS, 0);
    tarVec[0] = row.classification == 'g';
    tarVec[1] = row.classification == 'h';
}

/**
 * Runs the seed with the given datasets.
 * This function is supposed to be run in a thread.
 * @param seed the number of the seed to be run
 * @param trainingset set of training data
 * @param testingset set of testing data
 */
static void runSeed(const int seed, const vector<DatasetRow> &trainingset, const vector<DatasetRow> &testingset) {
    const string filename = "startnet_seed_" + to_string(seed) + ".net";
    const auto net = new Net();
    net->load_net(filename.c_str());
    const auto inVec = new float[net->topo_data.in_count];
    const auto outVec = new float[net->topo_data.out_count];
    const auto tarVec = new float[net->topo_data.out_count];

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
        float correctRate = ((float) (correct) / (float) (testingset.size())) * 100.0F;
        printf("%d/%d correct=%d/%zu %.3f%%\n", seed, epoch, correct, testingset.size(), correctRate);
    }
    delete[] inVec;
    delete[] outVec;
    delete[] tarVec;
}

int main(int argc, char **argv) {
    // Parse given command line arguments
    int partitionCount = 1;
    // Check argument count and correct command usage
    if (argc < 3 || argc > 4) {
        printf("<seed> = integer used as seed for the random generator\n");
        printf("<threadcount> = amount of threads to be used\n");
        printf("<partitionCount> = amount of threads to be used\n");
        printf("./lucavinciguerra-bathesis <seed> <threadcount>\n");
        printf("./lucavinciguerra-bathesis <seed> <threadcount> <partitionCount>\n");
        return 1;
    }
    // Parse given threadcount
    int randomSeed = stoi(argv[1]);
    int threadcount = stoi(argv[2]);
    if (argc == 4) {
        partitionCount = stoi(argv[3]);
    }
    // Read in dataset from the dataset textfile
    vector<DatasetRow> dataset;
    if (!readDataset(DATASET_FILENAME, &dataset, randomSeed)) {
        printf("Could not read dataset from file %s. Does the file exist?\n", DATASET_FILENAME);
        return 1;
    }
    printf("Read dataset from file %s with %zu rows.\n", DATASET_FILENAME, dataset.size());
    // vector that contains all threads that should run in parallel
    ThreadPool *threadpool = new ThreadPool(threadcount);
    for (int seed = 1; seed <= SEEDS; seed++) {
        // Partition the dataset into a training set and a testing set for this seed
        vector<DatasetRow> trainingset, testingset;
        partitionDataset(dataset, partitionCount, 1, &trainingset, &testingset);
        printf("starting thread for seed %d with %zu training and %zu testing rows\n", seed, trainingset.size(),
               testingset.size());
        // add the thread to the list of threads
        threadpool->enqueue([seed, trainingset, testingset] {
            runSeed(seed, trainingset, testingset);
        });
    }
    delete threadpool;
    return 0;
}
