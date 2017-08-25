#pragma once

#ifdef LiblinearCLASSIFIER_EXPORTS
#define LiblinearCLASSIFIER_API __declspec(dllexport)
#else
#define LiblinearCLASSIFIER_API __declspec(dllimport)
#endif

#include "linear.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::ifstream;

class FileReader
{
public:
	string fileName;
	ifstream fileReader;
	FileReader(const char* filename)
	{
		fileReader.open(fileName);
		if (!fileReader.is_open())
		{
			throw std::runtime_error("Can not find given file!!");
		}
		fileName = filename;
	}

	~FileReader()
	{
		if (fileReader.is_open())
		{
			fileReader.close();
		}
	}
};

class LiblinearTrainResource
{
public:
	parameter trainPara;
	problem trainProb;
	LiblinearTrainResource() {};
	~LiblinearTrainResource()
	{
		destroy_param(&trainPara);
		free(trainProb.y);
		free(trainProb.x);
	}
};

extern "C" LiblinearCLASSIFIER_API int LiblinearTrain(const char* modelFile,
	const char* dataFile,
	const char* configFile);

extern "C" LiblinearCLASSIFIER_API model* LiblinearLoad(const char* modelFile);
extern "C" LiblinearCLASSIFIER_API double LiblinearPredict(model* modelSol, int numOfFeature, int* indexList, double* valueList);
extern "C" LiblinearCLASSIFIER_API void LiblinearPredictAll(model* modelSol, int numOfFeature, int* indexList, double* valueList, double* res);
extern "C" LiblinearCLASSIFIER_API int LiblinearClassNum(model* model);
extern "C" LiblinearCLASSIFIER_API void LiblinearDelete(model* model);
extern "C" LiblinearCLASSIFIER_API void VectorDelete(void* ptr);
extern "C" LiblinearCLASSIFIER_API void* VectorCreate(int size);