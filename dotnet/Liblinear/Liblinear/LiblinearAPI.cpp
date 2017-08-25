#pragma once

#include "LiblinearAPI.h"
#include "linear.h"
#include <string.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <memory>

using std::vector;
using std::string;
using std::allocator;

model* LiblinearLoad(const char* modelFile)
{
	return load_model(modelFile);
}

int LiblinearTrain(const char* modelFile,
	const char* dataFile,
	const char* configFile)
{
	FileReader dataFileReader(dataFile);
	FileReader configFileReader(configFile);
	LiblinearTrainResource trainSrc;

	dataFileReader.fileReader >> trainSrc.trainProb.l;
	trainSrc.trainProb.y = (double*)malloc(sizeof(double)*trainSrc.trainProb.l);
	trainSrc.trainProb.x = (feature_node**)malloc(sizeof(feature_node*)*trainSrc.trainProb.l);
	dataFileReader.fileReader >> trainSrc.trainProb.bias;
	for (int i = 0; i < trainSrc.trainProb.l; i++)
	{
		int featureNum;
		dataFileReader.fileReader >> trainSrc.trainProb.y[i] >> featureNum;
		trainSrc.trainProb.x[i] = (feature_node*)malloc(sizeof(feature_node)*(featureNum + 1));
		for (int j = 0; j < featureNum; j++)
		{
			dataFileReader.fileReader >> trainSrc.trainProb.x[i][j].index >> trainSrc.trainProb.x[i][j].value;
		}
		trainSrc.trainProb.x[i][featureNum].index = -1;
	}
	dataFileReader.fileReader >> trainSrc.trainProb.n;

	configFileReader.fileReader >> trainSrc.trainPara.solver_type >> trainSrc.trainPara.eps >> trainSrc.trainPara.C >> trainSrc.trainPara.p;
	configFileReader.fileReader >> trainSrc.trainPara.nr_weight;
	if (trainSrc.trainPara.nr_weight > 0)
	{
		trainSrc.trainPara.weight_label = (int*)malloc(sizeof(int)*trainSrc.trainPara.nr_weight);
		trainSrc.trainPara.weight = (double*)malloc(sizeof(double)*trainSrc.trainPara.nr_weight);
		for (int i = 0; i < trainSrc.trainPara.nr_weight; i++)
		{
			configFileReader.fileReader >> trainSrc.trainPara.weight_label[i] >> trainSrc.trainPara.weight[i];
		}
	}
	else
	{
		trainSrc.trainPara.weight_label = NULL;
		trainSrc.trainPara.weight = NULL;
	}

	if (trainSrc.trainPara.solver_type != 0 && trainSrc.trainPara.solver_type != 2)
	{
		trainSrc.trainPara.init_sol = NULL;
	}

	const char* errMsg = check_parameter(&trainSrc.trainProb, &trainSrc.trainPara);
	if (errMsg != NULL)
	{
		throw std::runtime_error("Check parameter failed!!");
	}

	model* res = train(&trainSrc.trainProb, &trainSrc.trainPara);
	int classNum = res->nr_class;

	save_model(modelFile, res);
	free_and_destroy_model(&res);
	return classNum;
}

// Return the result of classification
double LiblinearPredict(model* modelSol, int numOfFeature, int* indexList, double* valueList)
{
	std::unique_ptr<feature_node[]> featureListPtr(new feature_node[numOfFeature + 1]);
	feature_node* featureList = featureListPtr.get();
	for (int i = 0; i < numOfFeature; i++)
	{
		featureList[i].index = indexList[i];
		featureList[i].value = valueList[i];
	}
	featureList[numOfFeature].index = -1;
	double res = predict(modelSol, featureList);
	return res;
}

//return the probability of every classes
void LiblinearPredictAll(model* modelSol, int numOfFeature, int* indexList, double* valueList, double* res)
{
	std::unique_ptr<feature_node[]> featureListPtr(new feature_node[numOfFeature + 1]);
	feature_node* featureList = featureListPtr.get();
	for (int i = 0; i < numOfFeature; i++)
	{
		featureList[i].index = indexList[i];
		featureList[i].value = valueList[i];
	}
	featureList[numOfFeature].index = -1;
	predict_values(modelSol, featureList, res);
}

int LiblinearClassNum(model* model)
{
	return model->nr_class;
}

void LiblinearDelete(model* model)
{
	free_and_destroy_model(&model);
}

void VectorDelete(void* ptr)
{
	free(ptr);
}

void* VectorCreate(int size)
{
	return malloc(size);
}