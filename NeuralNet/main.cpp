#include <vector>
#include "NetNN.h"

using namespace std;

void showVectorVals(string label, vector<double> &v)
{
	printf("%s ", label.c_str());
	for (unsigned i = 0; i < v.size(); ++i) {
		printf("%f ", v[i]);
	}
	printf("\n");
}
int main() {
	//e.g., {3, 2, 1 }
	vector<unsigned> topology;
	topology.push_back(2);
	topology.push_back(1);
	topology.push_back(1);

	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;


	for (int i = 0; i < 100000; i++) {
		double tout = rand() / (double)(RAND_MAX);

		inputVals.clear();
		inputVals = { 0, tout };

		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);

		targetVals.clear();
		targetVals = { tout };

	
		myNet.backProp(targetVals);
		
		/*
		// =======
		inputVals.clear();
		inputVals = { 0, 1 };

		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);

		targetVals.clear();
		targetVals = { 1 };

		myNet.backProp(targetVals);


		// =======
		inputVals.clear();
		inputVals = { 1, 0 };

		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);

		targetVals.clear();
		targetVals = { 0 };

		myNet.backProp(targetVals);


		// =======
		inputVals.clear();
		inputVals = { 1, 1 };

		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);

		targetVals.clear();
		targetVals = { 1 };

		myNet.backProp(targetVals);
		*/
		printf("Error: %f\n", myNet.getRecentAverageError());

	}

	inputVals.clear();
	inputVals = { 0, 0.7 };

	myNet.feedForward(inputVals);
	myNet.getResults(resultVals);

	showVectorVals("Outputs:", resultVals);

	std::system("pause");
}