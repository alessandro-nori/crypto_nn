#ifndef _relu_H
#define _relu_H

#include <cstdint>
#include "../../FHE.h"

vector<Ctxt> relu(vector<Ctxt> input);
vector<int64_t> relu(vector<int64_t> input);

int64_t get_scale();

#endif
