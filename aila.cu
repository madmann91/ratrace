// CudaTracerKernels.hpp
#pragma once
#include <cuda.h>
#include "helper_math.h"

//------------------------------------------------------------------------
// Constants.
//------------------------------------------------------------------------

enum
{
    MaxBlockHeight      = 6,            // Upper bound for blockDim.y.
    EntrypointSentinel  = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

#define TRACE_FUNC \
    extern "C" __global__ void trace( \
        int             numRays,        /* Total number of rays in the batch. */ \
        bool            anyHit,         /* False if rays need to find the closest hit. */ \
        float4*         rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*           results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*         nodesA,         /* SOA: bytes 0-15 of each node, AOS/Compact: 64 bytes per node. */ \
        float4*         nodesB,         /* SOA: bytes 16-31 of each node, AOS/Compact: unused. */ \
        float4*         nodesC,         /* SOA: bytes 32-47 of each node, AOS/Compact: unused. */ \
        float4*         nodesD,         /* SOA: bytes 48-63 of each node, AOS/Compact: unused. */ \
        float4*         trisA,          /* SOA: bytes 0-15 of each triangle, AOS: 64 bytes per triangle, Compact: 48 bytes per triangle. */ \
        float4*         trisB,          /* SOA: bytes 16-31 of each triangle, AOS/Compact: unused. */ \
        float4*         trisC,          /* SOA: bytes 32-47 of each triangle, AOS/Compact: unused. */ \
        int*            triIndices)     /* Triangle index remapping table. */

struct RayStruct
{
    float   idirx;  // 1.0f / ray.direction.x
    float   idiry;  // 1.0f / ray.direction.y
    float   idirz;  // 1.0f / ray.direction.z
    float   tmin;   // ray.tminq
    float   dummy;  // Padding to avoid bank conflicts.
};

//------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------

#define FETCH_GLOBAL(NAME, IDX, TYPE) ((const TYPE*)NAME)[IDX]
#define FETCH_TEXTURE(NAME, IDX, TYPE) tex1Dfetch(t_ ## NAME, IDX)
#define FETCH_ARRAY(NAME, IDX, TYPE) NAME[IDX]
#define STORE_RESULT(RAY, TRI, T, U, V) ((int4*)results)[(RAY)] = make_int4(TRI, __float_as_int(T), __float_as_int(U), __float_as_int(V))

//------------------------------------------------------------------------

#ifdef __CUDACC__

template <class T> __device__ __inline__ void swap(T& a,T& b)
{
    T t = a;
    a = b;
    b = t;
}

// Using video instructions
__device__ __inline__ int   min_min   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }

__device__ __inline__ float fmin_fmin (float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax (float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin (float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax (float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {	return fmax_fmax( fminf(a0,a1), fminf(b0,b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{	return fmin_fmin( fmaxf(a0,a1), fmaxf(b0,b1), fmax_fmin(c0, c1, d)); }

#endif

//------------------------------------------------------------------------
// kepler_dynamic_fetch.cu

#include "CudaTracerKernels.hpp"

#define STACK_SIZE              64          // Size of the traversal stack in local memory.
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays


extern "C" __device__ int g_warpCounter;    // Work counter for persistent threads.

TRACE_FUNC
{
    // Traversal stack in CUDA thread-local memory.

    int traversalStack[STACK_SIZE];
    traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

    // Live state during traversal, stored in registers.

    float   origx, origy, origz;            // Ray origin.
    char*   stackPtr;                       // Current position in traversal stack.
    int     leafAddr;                       // First postponed leaf, non-negative if none.
    //int     leafAddr2;                      // Second postponed leaf, non-negative if none.
    int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;                       // Triangle index of the closest intersection, -1 if none.
    float   hitT;                           // t-value of the closest intersection.
    float   hitU;
    float   hitV;
    float   tmin;
    int     rayidx;
    float   oodx;
    float   oody;
    float   oodz;
    float   dirx;
    float   diry;
    float   dirz;
    float   idirx;
    float   idiry;
    float   idirz;

	
    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.
    // Persistent threads: fetch and process rays in a loop.

    do
    {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        // Fetch new rays from the global pool using lane 0.

        const bool          terminated     = nodeAddr==EntrypointSentinel;
        const unsigned int  maskTerminated = __ballot(terminated);
        const int           numTerminated  = __popc(maskTerminated);
        const int           idxTerminated  = __popc(maskTerminated & ((1u<<tidx)-1));
    
        if(terminated)
        {
            if (idxTerminated == 0)
                rayBase = atomicAdd(&g_warpCounter, numTerminated);

            rayidx = rayBase + idxTerminated;
            if (rayidx >= numRays)
                break;

            // Fetch ray.

            float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
            float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
            origx = o.x;
            origy = o.y;
            origz = o.z;
            tmin  = o.w;
            dirx  = d.x;
            diry  = d.y;
            dirz  = d.z;
            hitT  = d.w;

             float ooeps = exp2f(-80.0f); // Avoid div by zero.
            idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            oodx  = origx * idirx;
            oody  = origy * idiry;
            oodz  = origz * idirz;

            // Setup traversal.

            stackPtr = (char*)&traversalStack[0];
            leafAddr = 0;   // No postponed leaf.
            //leafAddr2= 0;   // No postponed leaf.
            nodeAddr = 0;   // Start from the root.
            hitIndex = -1;  // No triangle intersected so far.

        }

        // Traversal loop.

        while(nodeAddr != EntrypointSentinel)
        {
            // Traverse internal nodes until all SIMD lanes have found a leaf.

//          while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
            while ((unsigned int)(nodeAddr) < (unsigned int)(EntrypointSentinel))   // functionally equivalent, but faster
            {
                // Fetch AABBs of the two child nodes.

                const float4 n0xy = tex1Dfetch(t_nodesA, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
                const float4 n1xy = tex1Dfetch(t_nodesA, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
                const float4 nz   = tex1Dfetch(t_nodesA, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
                      float4 tmp  = tex1Dfetch(t_nodesA, nodeAddr + 3); // child_index0, child_index1

                int2  cnodes= *(int2*)&tmp;

                // Intersect the ray against the child nodes.

                const float c0lox = n0xy.x * idirx - oodx;
                const float c0hix = n0xy.y * idirx - oodx;
                const float c0loy = n0xy.z * idiry - oody;
                const float c0hiy = n0xy.w * idiry - oody;
                const float c0loz = nz.x   * idirz - oodz;
                const float c0hiz = nz.y   * idirz - oodz;
                const float c1loz = nz.z   * idirz - oodz;
                const float c1hiz = nz.w   * idirz - oodz;
                const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                const float c1lox = n1xy.x * idirx - oodx;
                const float c1hix = n1xy.y * idirx - oodx;
                const float c1loy = n1xy.z * idiry - oody;
                const float c1hiy = n1xy.w * idiry - oody;
                const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
                const float c1max = spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

                bool swp = (c1min < c0min);

                bool traverseChild0 = (c0max >= c0min);
                bool traverseChild1 = (c1max >= c1min);

                // Neither child was intersected => pop stack.

                if (!traverseChild0 && !traverseChild1)
                {
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // Otherwise => fetch child pointers.

                else
                {
                    nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                    // Both children were intersected => push the farther one.

                    if (traverseChild0 && traverseChild1)
                    {
                        if (swp)
                            swap(nodeAddr, cnodes.y);
                        stackPtr += 4;
                        *(int*)stackPtr = cnodes.y;
                    }
                }

                // First leaf => postpone and continue traversal.

                if (nodeAddr < 0 && leafAddr  >= 0)     // Postpone max 1
//              if (nodeAddr < 0 && leafAddr2 >= 0)     // Postpone max 2
                {
                    //leafAddr2= leafAddr;          // postpone 2
                    leafAddr = nodeAddr;
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // All SIMD lanes have found a leaf? => process them.

                // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
                // tried everything with CUDA 4.2 but always got several redundant instructions.

                unsigned int mask;
                asm("{\n"
                    "   .reg .pred p;               \n"
                    "setp.ge.s32        p, %1, 0;   \n"
                    "vote.ballot.b32    %0,p;       \n"
                    "}"
                    : "=r"(mask)
                    : "r"(leafAddr));
                if(!mask)
                    break;

                //if(!__any(leafAddr >= 0))
                //    break;
            }

            // Process postponed leaf nodes.

            while (leafAddr < 0)
            {
            
            
            	
			    for (int triAddr = ~leafAddr;; triAddr += 3)
                {
                	
                	 // Tris in TEX (good to fetch as a single batch)
                    const float3 v00 = make_float3(tex1Dfetch(t_trisA, triAddr + 0));
                    const float3 v11 = make_float3(tex1Dfetch(t_trisA, triAddr + 1));
                    const float3 v22 = make_float3(tex1Dfetch(t_trisA, triAddr + 2));

                    // ------ Modified  version of the intersection routine -----------

                    // End marker (negative zero) => all triangles processed.
                    if (__float_as_int(v00.x) == 0x80000000)
                        break;

					const float3 e1 = v00 - v11;
                    const float3 e2 = v22 - v00;
                    const float3 n = cross(e1, e2);
                    const float3 dir = make_float3(dirx,diry,dirz);
                    const float3 org = make_float3(origx, origy, origz);
                    const float3 c = v00 - org;
                    const float3 r = cross(dir, c);
                    const float det = dot(n, dir);
                    const float abs_det = fabsf(det);

                    const float u = __int_as_float(__float_as_int(dot(r, e2)) ^ (__float_as_int(det) & 0x80000000));
                    bool mask = u >= 0.0f;
                    const float v = __int_as_float(__float_as_int(dot(r, e1)) ^ (__float_as_int(det) & 0x80000000));
                    mask &= v >= 0.0f;
                    const float w = abs_det - u - v;
                    mask &= w >= 0.0f;

                    if (mask) {
                        const float t = __int_as_float(__float_as_int(dot(n, c)) ^ (__float_as_int(det) & 0x80000000));
                        mask &= (t >= abs_det * tmin) & (t <= abs_det * hitT) & (det != 0.0f);
                        if (mask) {
                            const float inv_det = 1.0f / abs_det;
                            hitT = t * inv_det;
                            hitU = u * inv_det;
                            hitV = v * inv_det;
                            hitIndex = triAddr;
                            if (anyHit) {
                                nodeAddr = EntrypointSentinel;
                                break;
                            }
                        }
                    }
				
                } // triangle

                // Another leaf was postponed => process it as well.

//              if(leafAddr2<0) { leafAddr = leafAddr2; leafAddr2=0; } else     // postpone2
                {
                    leafAddr = nodeAddr;
                    if (nodeAddr < 0)
                    {
                        nodeAddr = *(int*)stackPtr;
                        stackPtr -= 4;
                    }
                }
            } // leaf

            // DYNAMIC FETCH

            if( __popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD )
                break;

        } // traversal

        // Remap intersected triangle index, and store the result.

        if (hitIndex == -1) { STORE_RESULT(rayidx, -1, hitT, hitU, hitV); }
        else                { STORE_RESULT(rayidx, FETCH_TEXTURE(triIndices, hitIndex, int), hitT, hitU, hitV); }

    } while(true);
}

int checkCudaError() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
		return 1;
    }
    return 0;
}

int getFuncSharedSize() {
    cudaFuncAttributes  	attr;
    cudaFuncGetAttributes(&attr, "trace");
    checkCudaError();
    return attr.sharedSizeBytes;
}

void resetKernel() {
    //Reset Warp Counter
    int reset=0;
    cudaMemcpyToSymbol(g_warpCounter, &reset, sizeof(int), 0, cudaMemcpyHostToDevice);
    checkCudaError();
}

void launchKernel(int2 			  gridSize,    \
				  int2 			  blockSize,   \
				  int             numRays,     \
                  int             numNodes,    \
                  int             numTris,     \
		          float4*         rays_h,      \
		          float4*         rays_d,      \
    			  int4*           results_h,   \
    			  int4*           results_d,   \
    			  float4*         nodesA_h,    \
    			  float4*         nodesA_d,    \
				  float4*         trisA_h,     \
				  float4*         trisA_d,     \
				  int*            triIndices_h \
                  int*            triIndices_d) {
    // Kernel invocation
    dim3 numBlocks(gridSize.x, gridSize.y);
    dim3 threadsPerBlock(blockSize.x, blockSize.y);

#define RAY_SIZE   (sizeof(float4) * 2)
#define HIT_SIZE   (sizeof(int4))
#define NODE_SIZE  (sizeof(float4) * 4)
#define TRI_SIZE   (sizeof(float4) * 3)
#define INDEX_SIZE (sizeof(int))

    // Copy data to device
    cuMemcpyHtoD((CUdeviceptr) rays_d,        rays_h,       numRays  * RAY_SIZE);
    cuMemcpyHtoD((CUdeviceptr) results_d,     results_h,    numRays  * HIT_SIZE);
    cuMemcpyHtoD((CUdeviceptr) nodesA_d,      nodesA_h,     numNodes * NODE_SIZE);
    cuMemcpyHtoD((CUdeviceptr) triA_d,        triA_h,       numTris  * TRI_SIZE);
    cuMemcpyHtoD((CUdeviceptr) triIndices_d,  triIndices_h, numTris  * INDEX_SIZE);
    
	trace<<<numBlocks, threadsPerBlock>>>(numRays, false, rays, results, nodesA, nodesB, nodesC, nodesD, trisA, trisB, trisC, triIndices);   
    cudaDeviceSynchronize();

    // Copy data from device
    cuMemcpyDtoH(results_h, (CUdeviceptr) results_d, numRays * HIT_SIZE);
}

//------------------------------------------------------------------------
