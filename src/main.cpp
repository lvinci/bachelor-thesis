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
#define DATASET_INITIAL_SHUFFLE_SEED 13267
#define SEEDS 10
#define EPOCHS 10000

#define PERCENT_TRAINING 0.8
#define PERCENT_TESTING (1.0 - PERCENT_TRAINING)
#define INPUTS 10
#define OUTPUTS 2
#define HIDDEN 9
#define LAYERS 4

// Results are stored here
double results[SEEDS + 2][EPOCHS + 1][3];
double combinedMSE[SEEDS + 2][EPOCHS + 1];
double combinedMSETesting[SEEDS + 2][EPOCHS + 1];
double combinedCorrects[SEEDS + 2][EPOCHS + 1];

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
 * Initializes the given neural net with the predefined topology.
 * @param net Net instance to be initialized
 * @param randomSeed seed used by the random number generator
 */
void createNeuralNet(Net *net, int seed) {
    // Create neural network topology
    int topology[LAYERS];
    topology[0] = INPUTS;
    topology[1] = HIDDEN;
    topology[2] = HIDDEN;
    topology[3] = OUTPUTS;
    // Create the layers and connect the weights
    net->create_layers(LAYERS, topology);
    net->connect_layers();
    net->set_seed(seed);
    net->init_weights(0, 0.5);
    // Initialize activation functions
    float uparams[10];
    uparams[0] = 0.001;
    uparams[1] = 0.95;
    uparams[2] = 0;
    net->set_layer_act_f(1, LOGISTIC);
    net->set_layer_act_f(2, LOGISTIC);
    net->set_layer_act_f(3, LOGISTIC);
    net->set_update_f(BP, uparams);
}

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
    for (float & i : row->input) {
        if (!getline(ss, substr, ';') || !(istringstream(substr) >> i)) {
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
 * Checks whether the file with the given filename exists.
 * @param name relative file path and name
 * @return true if the file exists, false otherwise
 */
inline bool fileExists(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

/**
 * Reads the dataset from the file with the given filename
 * @param filename name (relative path) of the file to be read
 * @param dataset vector to save the parsed dataset rows
 * @return true if the dataset could be read, false otherwise
 */
bool readDataset(const string& filename, vector<DatasetRow> *dataset) {
    ifstream file(filename);
    if (!file || !file.is_open()) {
        return false;
    }
    string line;
    while (getline(file, line)) {
        DatasetRow row;
        if (parseDatasetRow(line, &row)) dataset->push_back(row);
    }
    shuffle(dataset->begin(), dataset->end(), default_random_engine(DATASET_INITIAL_SHUFFLE_SEED));
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
 * @param seed seed to be used for random generation
 */
void
partitionDataset(const vector<DatasetRow> &dataset, const int numPartitions, const int partition,
                 vector<DatasetRow> *trainingset,
                 vector<DatasetRow> *testingset, const int seed) {
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
    // Shuffle the dataset for the seed
    shuffle(trainingset->begin(), trainingset->end(), default_random_engine(seed));
    shuffle(testingset->begin(), testingset->end(), default_random_engine(seed));
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
 * Squares the given number and returns the result.
 * This function is used to make the analyzeResults function more readable.
 * @param number number to be squared
 * @return the number squared
 */
inline double square(double number) {
    return number * number;
}

/**
 * Analyzes the results and saves the data to a textfile
 */
void analyzeResults(const int threadcount, const int partitionCount, const int attempt) {
    srand(time(nullptr));
    ofstream experimentalData(
            "results_attempt" + to_string(attempt) + "_" + to_string(partitionCount) + "partitions_" +
            to_string(threadcount) + "threads.txt");
    for (int i = 0; i <= EPOCHS; i++) {
        double tssVariance, tssDeviation, testTssVariance, testTssDeviation, correctsVariance, correctsDeviation;
        double mse = 0, mseTesting = 0, corrects = 0;
        for (int seed = 1; seed <= SEEDS; seed++) {
            mse += combinedMSE[seed][i];
            mseTesting += combinedMSETesting[seed][i];
            corrects += combinedCorrects[seed][i];
        }
        tssVariance = (square((results[1][i][0] - mse / SEEDS)) +
                       square((results[2][i][0] - mse / SEEDS)) +
                       square((results[3][i][0] - mse / SEEDS)) +
                       square((results[4][i][0] - mse / SEEDS)) +
                       square((results[5][i][0] - mse / SEEDS)) +
                       square((results[6][i][0] - mse / SEEDS)) +
                       square((results[7][i][0] - mse / SEEDS)) +
                       square((results[8][i][0] - mse / SEEDS)) +
                       square((results[9][i][0] - mse / SEEDS)) +
                       square((results[10][i][0] - mse / SEEDS))) / SEEDS;
        tssDeviation = sqrt(tssVariance);
        testTssVariance = (square((results[1][i][1] - mseTesting / SEEDS)) +
                           square((results[2][i][1] - mseTesting / SEEDS)) +
                           square((results[3][i][1] - mseTesting / SEEDS)) +
                           square((results[4][i][1] - mseTesting / SEEDS)) +
                           square((results[5][i][1] - mseTesting / SEEDS)) +
                           square((results[6][i][1] - mseTesting / SEEDS)) +
                           square((results[7][i][1] - mseTesting / SEEDS)) +
                           square((results[8][i][1] - mseTesting / SEEDS)) +
                           square((results[9][i][1] - mseTesting / SEEDS)) +
                           square((results[10][i][1] - mseTesting / SEEDS))) / SEEDS;
        testTssDeviation = sqrt(testTssVariance);
        correctsVariance = (square((results[1][i][2] - corrects / SEEDS)) +
                            square((results[2][i][2] - corrects / SEEDS)) +
                            square((results[3][i][2] - corrects / SEEDS)) +
                            square((results[4][i][2] - corrects / SEEDS)) +
                            square((results[5][i][2] - corrects / SEEDS)) +
                            square((results[6][i][2] - corrects / SEEDS)) +
                            square((results[7][i][2] - corrects / SEEDS)) +
                            square((results[8][i][2] - corrects / SEEDS)) +
                            square((results[9][i][2] - corrects / SEEDS)) +
                            square((results[10][i][2] - corrects / SEEDS))) / SEEDS;
        correctsDeviation = sqrt(correctsVariance);
        experimentalData << i << "," << mse / SEEDS << "," << tssDeviation << ","
                         << mseTesting / SEEDS << "," << testTssDeviation << ","
                         << corrects / SEEDS << "," << correctsDeviation << endl;
    }
}

/**
 * Runs the seed with the given datasets.
 * This function is supposed to be run in a thread.
 * @param seed the number of the seed to be run
 * @param trainingset set of training data
 * @param testingset set of testing data
 */
static void runSeed(const int seed, const vector<DatasetRow> &trainingset, const vector<DatasetRow> &testingset) {
    // Initialize the net object for the given seed.
    const string filename = "startnet_seed_" + to_string(seed) + ".net";
    const auto net = new Net();
    // If the startnet file exists, use the generated startnet, otherwise generate the net and save
    if (fileExists(filename)) {
        net->load_net(filename.c_str());
    } else {
        createNeuralNet(net, seed);
        net->save_net(filename);
    }
    // Create the vectors used as inputs, outputs and targets for use by propagation
    const auto inVec = new float[net->topo_data.in_count];
    const auto outVec = new float[net->topo_data.out_count];
    const auto tarVec = new float[net->topo_data.out_count];
    // Run the training and testing process EPOCHS amount of times
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Run the training for the whole training set
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
        // Test the neural network for the whole testing set and keep track of correct passes
        float tssTesting = 0, tssCorrect;
        int correct = 0;
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
        //
        const int datasetCount = testingset.size() + trainingset.size();
        results[seed][epoch][0] = (tss / (datasetCount * PERCENT_TRAINING));
        results[seed][epoch][1] = (tssTesting / (datasetCount * PERCENT_TESTING));
        results[seed][epoch][2] = (correct / (datasetCount * PERCENT_TESTING)) * 100;
        combinedMSE[seed][epoch] = (tss / (datasetCount * PERCENT_TRAINING));
        combinedMSETesting[seed][epoch] = (tssTesting / (datasetCount * PERCENT_TESTING));
        combinedCorrects[seed][epoch] = (correct / (datasetCount * PERCENT_TESTING)) * 100;
        // Update the weights of the neural network and print the result of the epoch
        net->update_weights();
        float correctRate = ((float) (correct) / (float) (testingset.size())) * 100.0F;
        printf("%s%d/%d correct=%d/%zu %.3f%%\n", (seed < 10) ? "0" : "", seed, epoch, correct, testingset.size(),
               correctRate);
    }
    delete[] inVec;
    delete[] outVec;
    delete[] tarVec;
}

int main(int argc, char **argv) {
    // Parse given command line arguments
    if (argc < 2 || argc > 4) {
        printf("<threadcount> = amount of threads to be used\n");
        printf("<partitionCount> = amount of threads to be used\n");
        printf("<attempt> = number of the attempt\n");
        printf("./lucavinciguerra-bathesis <threadcount>\n");
        printf("./lucavinciguerra-bathesis <threadcount> <partitionCount>\n");
        printf("./lucavinciguerra-bathesis <threadcount> <partitionCount> <attempt>\n");
        return 1;
    }
    int threadcount = stoi(argv[1]);
    int partitionCount = 1;
    int attempt = 1;
    if (argc >= 3) {
        partitionCount = stoi(argv[2]);
    }
    if (argc >= 4) {
        attempt = stoi(argv[3]);
    }
    // Read in dataset from the dataset textfile
    vector<DatasetRow> dataset;
    if (!readDataset(DATASET_FILENAME, &dataset)) {
        printf("Could not read dataset from file %s. Does the file exist?\n", DATASET_FILENAME);
        return 1;
    }
    printf("Read dataset from file %s with %zu rows.\n", DATASET_FILENAME, dataset.size());
    // vector that contains all threads that should run in parallel
    auto *threadpool = new ThreadPool(threadcount);
    for (int seed = 1; seed <= SEEDS; seed++) {
        // Partition the dataset into a training set and a testing set for this seed
        vector<DatasetRow> trainingset, testingset;
        partitionDataset(dataset, partitionCount, 1, &trainingset, &testingset, seed);
        printf("starting thread for seed %d with %zu training and %zu testing rows\n", seed, trainingset.size(),
               testingset.size());
        // add the thread to the list of threads
        threadpool->enqueue([seed, trainingset, testingset] {
            runSeed(seed, trainingset, testingset);
        });
    }
    delete threadpool;
    analyzeResults(threadcount, partitionCount, attempt);
    return 0;
}
