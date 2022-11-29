/*
<%
setup_pybind11(cfg)
%>
*/
#pragma warning(disable:4996)
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
//#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#define _CRT_SECURE_NO_WARNINGS

namespace py = pybind11;
#define item_num 64443   //64443  26813
#define user_num 75258   //75258  48799

double* item_popularity[item_num * sizeof(double)];
double* grad[item_num * sizeof(double)];
int* item_interaction[item_num * sizeof(int*)];
int* user_interaction[user_num * sizeof(int*)];

int idx, i, itr_num_item, j, itemid;
int timestamp;
int iteraction_num[item_num] = { 0 };
int iteraction_num_user[user_num] = { 0 };
double tau[item_num] = { 10000000 };//8000000;
double tmp = 0;

void load_popularity(py::array_t<double>& tau_p)
{
	//printf("%f", tau);
	FILE* fp;
	py::buffer_info buf1 = tau_p.request();
	auto tau_s = py::array_t<double>(buf1.size);
	double* ptr1 = (double*)buf1.ptr;
	
	//printf("%f", tau);

	char buffer[8192 * 16] = { 0 };
	char* line, * record = 0;

	if ((fp = fopen("./data/Amazon-CDs_and_Vinyl/item_interactions.csv", "r")) != NULL)                                  
	{ 
		
		for (i = 0; i < item_num; i++)
		{
			line = fgets(buffer, sizeof(buffer), fp);
			idx = atoi(strtok(line, ","));  //itemID
			tau[idx] = ptr1[idx];
			//printf("%d  %d\n", i, idx);
			if (idx != i)
			{
				printf("%d", idx);

			}
			itr_num_item = atoi(strtok(NULL, ","));
			iteraction_num[i] = itr_num_item;
			//printf("%d\n", itr_num_item);
			if (item_interaction[i] != NULL)
			{
				free(item_interaction[i]);
				item_interaction[i] = NULL;
			}
			if (item_popularity[i] != NULL)
			{
				free(item_popularity[i]);
				item_popularity[i] = NULL;
			}
			if (grad[i] != NULL)
			{
				free(grad[i]);
				grad[i] = NULL;
			}
			item_interaction[i] = (int*)malloc((itr_num_item) * sizeof(int));
			item_popularity[i] = (double*)malloc((itr_num_item) * sizeof(double));
			grad[i] = (double*)malloc((itr_num_item) * sizeof(double));
			for (j = 0; j < itr_num_item; j++)				//读入timestamp
			{
				timestamp = atoi(strtok(NULL, ","));
				item_interaction[i][j] = timestamp;
			}
			item_popularity[i][0] = 0;
			grad[i][0] = 0;
			for (j = 1; j < itr_num_item; j++)				//计算所有时刻的popularity和gradient
			{
				item_popularity[i][j] = (item_popularity[i][j - 1] + 1) * exp(-1 * double(item_interaction[i][j] - item_interaction[i][j - 1]) / tau[idx]);
				tmp = (item_interaction[i][j] - item_interaction[i][j - 1]) / tau[idx] / tau[idx];
				grad[i][j] = (grad[i][j - 1] + tmp) * exp(-1 * double(item_interaction[i][j] - item_interaction[i][j - 1]) / tau[idx]);
				//printf("%f\n", item_popularity[i][j]);
			}
		
		}
		//printf("load finished\n");
		
		fclose(fp);
		
	}
	else
	{
		printf("failed to open file!\n");
	}


}

py::array_t<double> popularity(py::array_t<double>& item_p, py::array_t<double>& timestamp_p)
{
	py::buffer_info buf1 = item_p.request();
	py::buffer_info buf2 = timestamp_p.request();

	auto popularity_result = py::array_t<double>(buf1.size);
	py::buffer_info buf3 = popularity_result.request();


	//获取numpy.ndarray 数据指针
	double* ptr1 = (double*)buf1.ptr;
	double* ptr2 = (double*)buf2.ptr;
	double* ptr3 = (double*)buf3.ptr;


	//指针访问numpy.ndarray
	for (int i = 0; i < buf1.shape[0]; i++)//
	{
		itemid = int(ptr1[i]);
		//printf("%d,%d\n", i, itemid);
		timestamp = int(ptr2[i]);
		if (iteraction_num[itemid] == 0)
		{
			ptr3[i] = 0;
			continue;
		}
		j = 0;
		while (j+100 < iteraction_num[itemid] && item_interaction[itemid][j + 100] <= timestamp)
		{
			j = j+100;
		}
		while (j+10 < iteraction_num[itemid] && item_interaction[itemid][j + 10] <= timestamp)
		{
			j = j+10;
		}
		for (; item_interaction[itemid][j] <= timestamp && j < iteraction_num[itemid];j++)
		{
			
		}
		if (j == 0)
		{
			ptr3[i] = 0;
		}
		else if (item_interaction[itemid][j-1] == timestamp)
		{
			ptr3[i] = item_popularity[itemid][j-1] + 1;
		}
		else
		{
			ptr3[i] = (item_popularity[itemid][j - 1] + 1) * exp(-1 * double(timestamp - item_interaction[itemid][j - 1]) / tau[itemid]);
		}
	}

	return popularity_result;
}

py::array_t<double> gradient(py::array_t<double>& item_p, py::array_t<double>& timestamp_p)
{
	py::buffer_info buf1 = item_p.request();
	py::buffer_info buf2 = timestamp_p.request();

	auto grad_result = py::array_t<double>(buf1.size);
	py::buffer_info buf4 = grad_result.request();

	//获取numpy.ndarray 数据指针
	double* ptr1 = (double*)buf1.ptr;
	double* ptr2 = (double*)buf2.ptr;
	double* ptr4 = (double*)buf4.ptr;

	//指针访问numpy.ndarray
	for (int i = 0; i < buf1.shape[0]; i++)//
	{
		itemid = int(ptr1[i]);
		//printf("%d,%d\n", i, itemid);
		timestamp = int(ptr2[i]);
		if (iteraction_num[itemid] == 0)
		{
			ptr4[i] = 0;
			continue;
		}
		for (j = 0; item_interaction[itemid][j] < timestamp && j < iteraction_num[itemid]; j++)
		{

		}
		tmp = (timestamp - item_interaction[itemid][j - 1]) / tau[itemid] / tau[itemid];
		ptr4[i] = (grad[itemid][j - 1] + tmp) * exp(-1 * double(timestamp - item_interaction[itemid][j - 1]) / tau[itemid]);
	}

	return grad_result;
}

void load_user_interation_val()
{
	FILE* fp;
	char buffer[8192 * 128] = { 0 };
	char* line = 0;
	int user_id = 0;
	int itr_num_user, item_id;
	// printf("loading user interaction\n");
	if ((fp = fopen("./data/Amazon-CDs_and_Vinyl/train_list.txt", "r")) != NULL)                                //*
	{
		while (!feof(fp))
		{
			line = fgets(buffer, sizeof(buffer), fp);
			user_id = atoi(strtok(line, "\t"));
			//printf("%d", user_id);
			itr_num_user = atoi(strtok(NULL, "\t"));
			iteraction_num_user[user_id] = itr_num_user;
			user_interaction[user_id] = (int*)malloc((itr_num_user) * sizeof(int));
			for (j = 0; j < itr_num_user; j++)				//读入用户交互历史 正样本
			{
				item_id = atoi(strtok(NULL, "\t"));
				user_interaction[user_id][j] = item_id;
			}
		}
		fclose(fp);
	}
}

void load_user_interation_test()
{
	FILE* fp;
	char buffer[8192 * 128] = { 0 };
	char* line = 0;
	int user_id = 0;
	int itr_num_user, item_id;
	// printf("loading user interaction\n");
	if ((fp = fopen("./data/Amazon-CDs_and_Vinyl/train_list_t.txt", "r")) != NULL)                         //*
	{
		while (!feof(fp))
		{
			line = fgets(buffer, sizeof(buffer), fp);
			user_id = atoi(strtok(line, "\t"));
			//printf("%d", user_id);
			itr_num_user = atoi(strtok(NULL, "\t"));
			iteraction_num_user[user_id] = itr_num_user;
			user_interaction[user_id] = (int*)malloc((itr_num_user) * sizeof(int));
			for (j = 0; j < itr_num_user; j++)				//读入用户交互历史 正样本
			{
				item_id = atoi(strtok(NULL, "\t"));
				user_interaction[user_id][j] = item_id;
			}
		}
		fclose(fp);
	}
}


py::array_t<double> negtive_sample(py::array_t<double>& user, py::array_t<double>& ng_num)
{
	py::buffer_info buf1 = user.request();
	py::buffer_info buf2 = ng_num.request();

	auto neg_items = py::array_t<double>(buf1.size * 4);
	py::buffer_info buf3 = neg_items.request();


	//获取numpy.ndarray 数据指针
	double* ptr1 = (double*)buf1.ptr;
	double* ptr2 = (double*)buf2.ptr;
	double* ptr3 = (double*)buf3.ptr;

	int nega_num = int(ptr2[0]);

	//指针访问numpy.ndarray
	int user_id, random_item;
	// printf("user_num = %d", int(buf1.shape[0]));
	for (int i = 0; i < int(buf1.shape[0]); i++)//
	{
		user_id = int(ptr1[i]);
		for (j = 0; j < nega_num;)
		{
			random_item = int(rand() % item_num);
			for (int n = 0; n < iteraction_num_user[user_id]; n++)  //直到找到正样本中不存在的item
			{
				if (random_item == user_interaction[user_id][n])
				{
					random_item = int(rand() % item_num);
					n = 0;
					continue;
				}
			}
			ptr3[nega_num * i + j] = random_item;
			j++;
		}
	}
	return neg_items;
}

PYBIND11_MODULE(pybind_amazon_cd, m) {

	m.doc() = "pybind11 example module";

	// Add bindings here

	m.def("load_popularity", &load_popularity);

	m.def("popularity", &popularity);

	m.def("gradient", &gradient);

	m.def("load_user_interation_val", &load_user_interation_val);

	m.def("load_user_interation_test", &load_user_interation_test);

	m.def("negtive_sample", &negtive_sample);

	m.def("foo", []() {
		return "Hello, World!";
		});

	m.def("foo2", []() {
		return "This is foo2!\n";
		});

	m.def("add", [](int a, int b) {
		return a + b;
		});

	m.def("sub", [](int a, int b) {
		return a - b;
		});

	m.def("mul", [](int a, int b) {
		return a * b;
		});

	m.def("div", [](int a, int b) {
		return static_cast<float>(a) / b;
		});

}

