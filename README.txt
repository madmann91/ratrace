This archive contains the source for our traversal in three files:
* common.impala contains the generic parts of the traversal
* mapping_cpu.impala and mapping_gpu.impala contain the target specific mappings
These files are distributed under the LGPL license.

We also provide the excerpts from Embree and the work of Aila et al. that we used to measure code complexity: they can be found in the files aila.cu and embree.cpp.
These files only mention the parts that are relevant for our paper, and are under the license of their respective authors.
