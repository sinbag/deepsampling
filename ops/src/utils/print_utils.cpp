#include "print_utils.h"

//===========================================

void print(string output)
{
	cout << preStr << output << endl;
}

//===========================================

void print(float output)
{
	cout << preStr << to_string(output) << endl;
}

//===========================================

void print(int output)
{
	cout << preStr << to_string(output) << endl;
}

//===========================================

void print(const Eigen::VectorXf& v)
{
	cout << preStr << "[";
	for (int i=0; i<v.size(); i++)
	{
		cout << v(i);
		if (i<v.size()-1) cout << ", ";
	}
	cout << "]" << endl;
}