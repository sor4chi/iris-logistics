#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

string fp = "./data/iris.csv";

struct Iris
{
  double sepal_length;
  double sepal_width;
  double petal_length;
  double petal_width;
  string species;
};

vector<Iris> read_csv(string fp)
{
  vector<Iris> data;
  ifstream ifs(fp);
  string line;

  // skip header
  getline(ifs, line);

  // load data
  while (getline(ifs, line))
  {
    Iris iris;
    string buf;
    stringstream ss(line);
    getline(ss, buf, ',');
    iris.sepal_length = stod(buf);
    getline(ss, buf, ',');
    iris.sepal_width = stod(buf);
    getline(ss, buf, ',');
    iris.petal_length = stod(buf);
    getline(ss, buf, ',');
    iris.petal_width = stod(buf);
    getline(ss, buf, ',');
    iris.species = buf;
    data.push_back(iris);
  }
  return data;
}

double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

double loss(vector<Iris> data, double w1, double w2, double w3, double w4, double b)
{
  double loss = 0.0;
  for (int i = 0; i < data.size(); i++)
  {
    double x = data[i].sepal_length;
    double y = data[i].sepal_width;
    double z = data[i].petal_length;
    double w = data[i].petal_width;
    double t = data[i].species == "Iris-setosa" ? 1.0 : 0.0;
    double y_ = sigmoid(w1 * x + w2 * y + w3 * z + w4 * w + b);
    double log_y_ = log(y_) == 0 ? 1e-10 : log(y_);
    double log_1_y_ = log(1 - y_) == -INFINITY ? -1e-10 : log(1 - y_);
    loss += -t * log_y_ - (1 - t) * log_1_y_;
  }
  return loss;
}

string fill_space(string s, int n)
{
  string res = s;
  for (int i = 0; i < n - s.size(); i++)
  {
    res += " ";
  }
  return res;
}

int main()
{
  vector<Iris> data = read_csv(fp);

  double w1 = 0.0;
  double w2 = 0.0;
  double w3 = 0.0;
  double w4 = 0.0;
  double b = 0.0;
  double lr = 0.1;
  double l = 0.0;
  for (int i = 0; i < 1000; i++)
  {
    double w1_grad = 0.0;
    double w2_grad = 0.0;
    double w3_grad = 0.0;
    double w4_grad = 0.0;
    double b_grad = 0.0;
    for (int j = 0; j < data.size(); j++)
    {
      double x = data[j].sepal_length;
      double y = data[j].sepal_width;
      double z = data[j].petal_length;
      double w = data[j].petal_width;
      double t = data[j].species == "Iris-setosa" ? 1.0 : 0.0;
      double y_ = sigmoid(w1 * x + w2 * y + w3 * z + w4 * w + b);
      w1_grad += (y_ - t) * x;
      w2_grad += (y_ - t) * y;
      w3_grad += (y_ - t) * z;
      w4_grad += (y_ - t) * w;
      b_grad += (y_ - t);
    }
    w1 -= lr * w1_grad;
    w2 -= lr * w2_grad;
    w3 -= lr * w3_grad;
    w4 -= lr * w4_grad;
    b -= lr * b_grad;
    l = loss(data, w1, w2, w3, w4, b);
    cout << "loss: " << l << endl;
  }
  cout << "w1: " << w1 << endl;
  cout << "w2: " << w2 << endl;
  cout << "w3: " << w3 << endl;
  cout << "w4: " << w4 << endl;
  cout << "b: " << b << endl;
  return 0;
}
