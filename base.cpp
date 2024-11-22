#include <iostream>
#include <cstdlib>

using namespace std;

int main() {
    // string username;
    // int userage;

    // cout << "yoooo!" << endl;
    // cout << "please enter your name: " << endl;
    // cin >> username;
    // cout << "please enter your age too :D " << endl;
    // cin >> userage;

    // cout << "here is your information, name: " << username << ", and age: " << userage << endl;

    const int size = 10;
    int arr1[size];
    int arr2[size];
    int sumList[size];

    srand(32);

    for(int i = 0; i < size; ++i) {
        arr1[i] = rand() % 100;
        arr2[i] = rand() % 100;
    }

    for(int i = 0; i < size; ++i) {
        sumList[i] = arr1[i] + arr2[i];
    }
    
    cout << "List 1: ";
    for(int i=0; i<size; ++i) {
        cout << arr1[i] << " ";
    }
    cout << endl;

    cout << "List 2: ";
    for(int i=0; i<size; ++i) {
        cout << arr2[i] << " ";
    }
    cout << endl;
    cout << "Sum: ";
    for(int i=0; i<size; ++i) {
        cout << sumList[i] << " ";
    }

    return 0;
}