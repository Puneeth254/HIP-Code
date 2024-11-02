#ifndef GENHIP_CONVEXHULL_H
#define GENHIP_CONVEXHULL_H

#include <iostream>
#include <climits>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "../graph.hpp"

auto extremePoints(
  std::vector<double>&  xCoord,std::vector<double>&  yCoord);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
auto innerPoint(
  std::vector<int>&  R,std::vector<double>&  xCoord,std::vector<double>&  yCoord);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
void getSign(
  std::vector<int>&  R,std::vector<int>&  sign,std::vector<double>&  point,std::vector<double>&  xCoord,
  std::vector<double>&  yCoord,std::vector<int>&  label,std::vector<int>&  L,std::vector<int>&  mark,
  int  maxLabel);


// DEVICE ASSTMENT in .h
__global__ void getSign_kernel0(int V, int* L, int* mark, int* R, int* sign, double* xCoord, double* yCoord, double* point, int maxLabel, int* label){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;

  if (mark[v] == 0) {
    int lab = label[v];  // DEVICE ASSTMENT in .h
    int left = R[lab];  // DEVICE ASSTMENT in .h
    lab = lab + 1;

    if (lab == maxLabel) {
      lab = 0;

    } 
    int right = R[lab];  // DEVICE ASSTMENT in .h
    int z = 0;  // DEVICE ASSTMENT in .h
    int o = 1;  // DEVICE ASSTMENT in .h
    double dist = (xCoord[right] - xCoord[left]) * (point[o] - yCoord[left]) - (yCoord[right] - yCoord[left]) * (point[z] - xCoord[left]);  // DEVICE ASSTMENT in .h

    if (dist <= 0) {
      sign[lab] = -1;

    }  else {
      sign[lab] = 1;

    }

  } 
} // end KER FUNC

void updateDistance(
  std::vector<int>&  label,std::vector<float>&  distance,std::vector<double>&  xCoord,std::vector<double>&  yCoord,
  std::vector<int>&  L,std::vector<int>&  R,std::vector<int>&  mark,std::vector<int>&  sign,
  int  maxLabel);


// DEVICE ASSTMENT in .h
__global__ void updateDistance_kernel1(int V, int* L, int* mark, float* distance, int* sign, double* yCoord, int maxLabel, double* xCoord, int* R, int* label){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;

  if (mark[v] == 0) {
    int lab = label[v];  // DEVICE ASSTMENT in .h
    int left = R[lab];  // DEVICE ASSTMENT in .h
    lab = lab + 1;

    if (lab == maxLabel) {
      lab = 0;

    } 
    int right = R[lab];  // DEVICE ASSTMENT in .h
    double dist = (xCoord[right] - xCoord[left]) * (yCoord[v] - yCoord[left]) - (yCoord[right] - yCoord[left]) * (xCoord[v] - xCoord[left]);  // DEVICE ASSTMENT in .h

    if (sign[lab] == -1) {

      if (dist <= 0) {
        mark[v] = 1;

      } 

    }  else {

      if (sign[lab] == 1) {

        if (dist >= 0) {
          mark[v] = 1;

        } 
        dist = -1 * dist;

      } 
    }
    distance[v] = dist;

  } 
} // end KER FUNC

auto updateHull(
  std::vector<float>&  distance,std::vector<double>&  xCoord,std::vector<double>&  yCoord,std::vector<int>&  R,
  std::vector<int>&  label,std::vector<int>&  mark,std::vector<int>&  sign,std::vector<double>&  point,
  std::vector<int>&  L,int  maxLabel);


// DEVICE ASSTMENT in .h
__global__ void updateHull_kernel2(int V, int n, int* R){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;

  if (v < n) {
    R[v + 2 * n] = -1;

  } 
} // end KER FUNC

// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
__global__ void updateHull_kernel3(int V, int* newR, int* R, int n){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  int temp = v - v / 2;  // DEVICE ASSTMENT in .h

  if (temp == v / 2) {
    int index = v / 2;  // DEVICE ASSTMENT in .h
    newR[v] = R[index];

  }  else {

    if (v / 2 < n) {
      int index = v / 2 + 2 * n;  // DEVICE ASSTMENT in .h
      newR[v] = R[index];

    } 
  }
} // end KER FUNC

__global__ void updateHull_kernel4(int V, int* R, int* newR){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  R[v] = newR[v];
} // end KER FUNC

void updateLabel(
  std::vector<int>&  label,std::vector<int>&  R,std::vector<double>&  xCoord,std::vector<double>&  yCoord,
  std::vector<int>&  mark,int  currHullSize);


// DEVICE ASSTMENT in .h
__global__ void updateLabel_kernel5(int V, int* mark, double* yCoord, double* xCoord, int* R, int currHullSize, int* label){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;

  if (mark[v] == 0) {
    int left = 2 * label[v];  // DEVICE ASSTMENT in .h
    int right = left + 2;  // DEVICE ASSTMENT in .h

    if (right >= currHullSize) {
      right = 0;

    } 
    left = R[left];
    right = R[right];
    int x = xCoord[left];  // DEVICE ASSTMENT in .h
    int y = yCoord[left];  // DEVICE ASSTMENT in .h
    int px = xCoord[v];  // DEVICE ASSTMENT in .h
    int py = yCoord[v];  // DEVICE ASSTMENT in .h
    double dist1 = (x - px) * (x - px) + (y - py) * (y - py);  // DEVICE ASSTMENT in .h
    x = xCoord[right];
    y = yCoord[right];
    double dist2 = (x - px) * (x - px) + (y - py) * (y - py);  // DEVICE ASSTMENT in .h

    if (dist2 < dist1) {
      label[v] = 2 * label[v] + 1;

    }  else {
      label[v] = 2 * label[v];

    }

  } 
} // end KER FUNC

auto convexHull(
  int  n,std::vector<double>&  xCoord,std::vector<double>&  yCoord);


// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
__global__ void convexHull_kernel6(int V, int* L){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  L[v] = v;
} // end KER FUNC

// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h
// DEVICE ASSTMENT in .h

#endif
