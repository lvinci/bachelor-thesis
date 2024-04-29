//============================================================================
// Name        : Biodegradation.cpp
// Author      : Artur Brening
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include "n++.h"
using namespace std;

#define PERCENT_TRAINING 0.8
#define PERCENT_TESTING 0.2
#define INPUTS 10
#define OUTPUTS 2
#define HIDDEN 9
#define LAYERS 4

Net* net;
string* originalData;	// pointer to data-Array
string* tempData;
string* trainingData;
string* testingData;
int dataSetCount;
int seed;
int seedCounter;

int topology[LAYERS];
float uparams[10];
FTYPE *in_vec, *out_vec, *tar_vec;

void CreateNet(Net* net, int inputs, int hidden, int outputs)
{
	topology[0] = inputs; topology[1] = hidden; topology[2] = hidden; topology[3] = outputs;

	net->create_layers(LAYERS, topology);
	net->connect_layers();
	net->init_weights(0, 0.5);

	uparams[0] = 0.001;
	uparams[1] = 0.95;
	uparams[2] = 0;

	net->set_layer_act_f(1, LOGISTIC);			// set hidden layer activation function to logistic
	net->set_layer_act_f(2, LOGISTIC);			// set hidden layer activation function to logistic
	net->set_layer_act_f(3, LOGISTIC);			// set output layer activation function to logistic
	net->set_update_f(BP, uparams);				// use backpropagation as learning mechanism

	in_vec = new float[net->topo_data.in_count];
	out_vec = new float[net->topo_data.out_count];
	tar_vec = new float[net->topo_data.out_count];
}

void InitializeVectors(char* cString)
{
	char* pch;
	int i = 0;
	pch = strtok(cString, ";");

	do
	{
		in_vec[i] = atof(pch);
		pch = strtok(NULL, ";");

		i++;
	}while(i < INPUTS);

	if (strcmp(pch, "g") == 0)
	{
		tar_vec[0] = 1;
		tar_vec[1] = 0;
	} else if (strcmp(pch, "h") == 0)
	{
		tar_vec[0] = 0;
		tar_vec[1] = 1;
	}

	return;
}

int CountLines(string fileName)
{
	ifstream infile(fileName);
	return count(istreambuf_iterator<char>(infile), istreambuf_iterator<char>(), '\n');	// counts number of lines in text-file
}

void InitializeDataSets(string* data, char* delimiter)
{
	trainingData = new string[(int)(dataSetCount*PERCENT_TRAINING)];
	testingData = new string[(int)(dataSetCount*PERCENT_TESTING)];

	float percentH = 0.352;
	float percentG = 0.648;

	int numH = nearbyint((dataSetCount*PERCENT_TRAINING)*percentH);
	int numG = nearbyint((dataSetCount*PERCENT_TRAINING)*percentG);

	int i;
	int hCounter;
	int gCounter;
	char* cString;
	char* pch;

	do
	{
		shuffle(data, data + dataSetCount, default_random_engine(seed));

		hCounter = 0;
		gCounter = 0;

		for(i = 0; i < dataSetCount*PERCENT_TRAINING; i++)
		{
			trainingData[i] = data[i];

			cString = new char[trainingData[i].size() + 1];
			strcpy(cString, trainingData[i].c_str());

			pch = strtok(cString, delimiter);

			while ((strcmp(pch, "g") != 0 && strcmp(pch, "h") != 0))
			{
				pch = strtok(NULL, delimiter);
			}

			if (strcmp(pch, "g") == 0)
				gCounter++;
			else if (strcmp(pch, "h") == 0)
				hCounter++;

			delete cString;
		}

		seed++;

	} while (hCounter != numH && gCounter != numG);

	cout << "Training-Data: " << endl;
	for(i = 0; i < dataSetCount*PERCENT_TRAINING; i++)
	{
		trainingData[i] = data[i];
		cout << trainingData[i] << endl;
	}

	cout << "Testing-Data: " << endl;
	for(int j = 0; i < dataSetCount; i++)
	{
		testingData[j] = data[i];
		cout << testingData[j] << endl;
		j++;
	}

	seedCounter++;
}

string* ReadDataToArray(string fileName)
{
	ifstream infile(fileName);
	string line;
	int i = 0;

	dataSetCount = CountLines(fileName);

	string* tmpArray = new string[dataSetCount];

	while(getline(infile, line))
	{
		istringstream iss(line);
		if (!(iss >> tmpArray[i])) { break; }	// error
		i++;
	}

	return tmpArray;
}

string* CopyData(string* original, int size)
{
	string* copy = new string[size];
	for(int i = 0; i < size; i++)
		copy[i] = original[i];

	return copy;
}

double square(double number)
{
	return number*number;
}

int main() {

	int maxEpoch = 10000;
	int correct;

	double combinedMSE[maxEpoch + 1];
	double combinedMSETesting[maxEpoch + 1];
	double combinedCorrects[maxEpoch + 1];
	double seed1[maxEpoch + 1][3];
	double seed2[maxEpoch + 1][3];
	double seed3[maxEpoch + 1][3];
	double seed4[maxEpoch + 1][3];
	double seed5[maxEpoch + 1][3];
	double seed6[maxEpoch + 1][3];
	double seed7[maxEpoch + 1][3];
	double seed8[maxEpoch + 1][3];
	double seed9[maxEpoch + 1][3];
	double seed10[maxEpoch + 1][3];

	uparams[0] = 0.001;
	uparams[1] = 0.95;
	uparams[2] = 0;

	string fileName;
	char* fileNamePointer;
	char delimiter = ';';
	char* cString;
	float error, tss, tssTesting, correctTss;

	originalData = ReadDataToArray("magic_normalized.txt");

	ofstream experimentalData;

	for(seed = 1; seedCounter < 10; seed++)
	{
		tempData = CopyData(originalData, dataSetCount);
		InitializeDataSets(tempData, &delimiter);

		//OutputDataSets();

		fileName = "startnet_seed_" + to_string(seed) + ".net";
		fileNamePointer = new char[fileName.size() + 1];
		copy(fileName.begin(), fileName.end(), fileNamePointer);
		fileNamePointer[fileName.size()] = '\0';

		net = new Net();
		net->load_net(fileNamePointer);

		delete fileNamePointer;

		in_vec = new float[net->topo_data.in_count];
		out_vec = new float[net->topo_data.out_count];
		tar_vec = new float[net->topo_data.out_count];

		net->save_net("start_uparam1_" + to_string(uparams[0]) + "_uparam2_" + to_string(uparams[1]) + "_uparam3_" + to_string(uparams[2]) + "_seed_" + to_string(seed) +".net");
		experimentalData.open("experimentalData_maxEpoch_" + to_string(maxEpoch) + "_uparam1_" + to_string(uparams[0]) + "_uparam2_" + to_string(uparams[1]) + "_uparam3_" + to_string(uparams[2]) + "_seed_" + to_string(seed) + ".txt");
		//experimentalData << "epoch,tss,correct" << endl;

		for (int epoch = 0; epoch <= maxEpoch; epoch++)
		{
			tss = 0.0;

			for (int counter = 0; counter < dataSetCount*PERCENT_TRAINING; counter++)
			{
				//cout << counter+1 << ". Training-Data: " << trainingData[counter] << endl;

				cString = new char[trainingData[counter].size() + 1];

				strcpy(cString, trainingData[counter].c_str());
				InitializeVectors(cString);

				delete cString;

				/*
				cout << "Input-Vector:" << endl;
				for(int i = 0; i < INPUTS; i++)
					cout << in_vec[i] << endl;
				cout << "Target-Vector:" << endl;
				for(int i = 0; i < OUTPUTS; i++)
					cout << tar_vec[i] << endl;
				*/

				net->forward_pass(in_vec, out_vec);	// calculate output into out_vec
				/*
				for(int i = 0; i < OUTPUTS; i++)
					printf("Output of Output-Node %d: %f\n", i+1, out_vec[i]);
				*/
				for(int i = 0; i < net->topo_data.out_count; i++)
				{
					error = out_vec[i] = out_vec[i] - tar_vec[i];		// dE_total/dout_oi
					tss += error*error;									// E_total
				}

				net->backward_pass(out_vec, in_vec);
				cout << "Epoch: " << epoch << ", TrainingSet: " << counter+1 << ", tss: " << (tss/(dataSetCount*PERCENT_TRAINING)) << endl;
			}

			correct = 0;
			tssTesting = 0;

			for (int counter = 0; counter < dataSetCount*PERCENT_TESTING; counter++)
			{
				correctTss = 0;

				//cout << "Testing-Data: " << testingData[counter] << endl;

				cString = new char[testingData[counter].size() + 1];

				strcpy(cString, testingData[counter].c_str());
				InitializeVectors(cString);

				delete cString;

				/*
				cout << "Input-Vector:" << endl;
				for(int i = 0; i < INPUTS; i++)
					cout << in_vec[i] << endl;
				cout << "Target-Vector:" << endl;
				for(int i = 0; i < OUTPUTS; i++)
					cout << tar_vec[i] << endl;
				*/

				net->forward_pass(in_vec, out_vec);

				for(int i = 0; i < net->topo_data.out_count; i++)
				{
					error = out_vec[i] - tar_vec[i];			// dE_total/dout_oi
					correctTss += error*error;							// E_total
					tssTesting += correctTss;
				}

				//cout << "tss: " << correctTss << endl;

				if (correctTss < 0.05)
					correct++;

				/*
				for(int i = 0; i < OUTPUTS; i++)
					printf("Output of Output-Node %d: %f\n", i+1, out_vec[i]);
				*/
			}

			//cout << correct << " out of " << dataSetCount*PERCENT_TESTING << " Testing-Datasets were correctly identified." << endl;
			experimentalData << epoch << "," << (tss/(dataSetCount*PERCENT_TRAINING)) << "," << (tssTesting/(dataSetCount*PERCENT_TESTING)) << "," << (correct/(dataSetCount*PERCENT_TESTING))*100 << endl;

			switch(seedCounter)
			{
				case 1: seed1[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed1[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed1[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 2: seed2[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed2[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed2[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 3: seed3[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed3[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed3[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 4: seed4[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed4[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed4[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 5: seed5[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed5[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed5[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 6: seed6[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed6[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed6[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 7: seed7[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed7[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed7[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 8: seed8[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed8[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed8[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 9: seed9[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						seed9[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						seed9[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						break;
				case 10: seed10[epoch][0] = (tss/(dataSetCount*PERCENT_TRAINING));
						 seed10[epoch][1] = (tssTesting/(dataSetCount*PERCENT_TESTING));
						 seed10[epoch][2] = (correct/(dataSetCount*PERCENT_TESTING))*100;
						 break;
			}

			combinedMSE[epoch] += (tss/(dataSetCount*PERCENT_TRAINING));
			combinedMSETesting[epoch] += (tssTesting/(dataSetCount*PERCENT_TESTING));
			combinedCorrects[epoch] += (correct/(dataSetCount*PERCENT_TESTING))*100;

			net->update_weights();
		}

		net->save_net("end_uparam1_" + to_string(uparams[0]) + "_uparam2_" + to_string(uparams[1]) + "_uparam3_" + to_string(uparams[2]) + "_seed_" + to_string(seed) + ".net");

		delete in_vec; delete out_vec; delete tar_vec;
		delete[] tempData;
		delete net;

		experimentalData.close();
	}

	double tssVariance;
	double tssDeviation;
	double testTssVariance;
	double testTssDeviation;
	double correctsVariance;
	double correctsDeviation;

	experimentalData.open("experimentalData_maxEpoch_" + to_string(maxEpoch) + "_uparam1_" + to_string(uparams[0]) + "_uparam2_" + to_string(uparams[1]) + "_uparam3_" + to_string(uparams[2]) + "_combinedSeeds.txt");

	for (int i = 0; i <= maxEpoch; i++)
	{
		tssVariance = 0;
		tssDeviation = 0;
		testTssVariance = 0;
		testTssDeviation = 0;
		correctsVariance = 0;
		correctsDeviation = 0;

		tssVariance = (square((seed1[i][0] - combinedMSE[i]/seedCounter)) + square((seed2[i][0] - combinedMSE[i]/seedCounter)) +
					  square((seed3[i][0] - combinedMSE[i]/seedCounter)) + square((seed4[i][0] - combinedMSE[i]/seedCounter)) +
					  square((seed5[i][0] - combinedMSE[i]/seedCounter)) + square((seed6[i][0] - combinedMSE[i]/seedCounter)) +
					  square((seed7[i][0] - combinedMSE[i]/seedCounter)) + square((seed8[i][0] - combinedMSE[i]/seedCounter)) +
					  square((seed9[i][0] - combinedMSE[i]/seedCounter)) + square((seed10[i][0] - combinedMSE[i]/seedCounter)))/seedCounter;

		tssDeviation = sqrt(tssVariance);

		testTssVariance = (square((seed1[i][1] - combinedMSETesting[i]/seedCounter)) + square((seed2[i][1] - combinedMSETesting[i]/seedCounter)) +
						  square((seed3[i][1] - combinedMSETesting[i]/seedCounter)) + square((seed4[i][1] - combinedMSETesting[i]/seedCounter)) +
						  square((seed5[i][1] - combinedMSETesting[i]/seedCounter)) + square((seed6[i][1] - combinedMSETesting[i]/seedCounter)) +
						  square((seed7[i][1] - combinedMSETesting[i]/seedCounter)) + square((seed8[i][1] - combinedMSETesting[i]/seedCounter)) +
						  square((seed9[i][1] - combinedMSETesting[i]/seedCounter)) + square((seed10[i][1] - combinedMSETesting[i]/seedCounter)))/seedCounter;

		testTssDeviation = sqrt(testTssVariance);

		correctsVariance = (square((seed1[i][2] - combinedCorrects[i]/seedCounter)) + square((seed2[i][2] - combinedCorrects[i]/seedCounter)) +
						  square((seed3[i][2] - combinedCorrects[i]/seedCounter)) + square((seed4[i][2] - combinedCorrects[i]/seedCounter)) +
						  square((seed5[i][2] - combinedCorrects[i]/seedCounter)) + square((seed6[i][2] - combinedCorrects[i]/seedCounter)) +
						  square((seed7[i][2] - combinedCorrects[i]/seedCounter)) + square((seed8[i][2] - combinedCorrects[i]/seedCounter)) +
						  square((seed9[i][2] - combinedCorrects[i]/seedCounter)) + square((seed10[i][2] - combinedCorrects[i]/seedCounter)))/seedCounter;

		correctsDeviation = sqrt(correctsVariance);

		experimentalData << i << "," << combinedMSE[i]/seedCounter << "," << tssDeviation << "," << combinedMSETesting[i]/seedCounter <<  "," <<  testTssDeviation << "," << combinedCorrects[i]/seedCounter << "," << correctsDeviation << endl;
	}

	experimentalData.close();

	return 0;
}
