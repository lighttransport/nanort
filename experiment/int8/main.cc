#include <cstdlib>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <type_traits>

#if defined(__AVX__)
#include <immintrin.h>
#endif

// From bitsandbytes ----------

#define BLOCK_SIZE 16384

struct quantize_block_args {
    //BinAlgo<Scalar, float, Direct2> *bin_searcher;
    float *code;
    float *A;
    float *absmax;
    unsigned char *out;
    long long block_end;
    long long block_idx;
    long long threadidx;
    long long blocksize;
};


void *quantize_block(void *arguments);


void *quantize_block(void *arguments) {
    // 1. find absmax in block
    // 2. divide input value by absmax to normalize into [-1.0, 1.0]
    // 3. do binary search to find the closest value
    // 4. check minimal distance
    // 5. store index

    struct quantize_block_args *args = (quantize_block_args *) arguments;

    // 1. find absmax in block
    float absmax_block = -(std::numeric_limits<float>::max)();
    for (long long i = args->block_idx; i < args->block_end; i++)
        absmax_block = (std::max)(absmax_block, std::abs(args->A[i]));

    args->absmax[args->block_idx / args->blocksize] = absmax_block;

    for (long long i = args->block_idx; i < args->block_end; i++) {
        // 2. divide input value by absmax to normalize into [-1.0, 1.0]
        // 3. do binary search to find the closest value
        float normed_value = args->A[i] / absmax_block;
        // TODO
        //long long idx = args->bin_searcher->scalar(normed_value);
        long long idx = 0;

        // 4. check minimal distance
        // The binary search returns always the value to the left, which might not be the closest value
        if (idx < 255) {
            float dist_left = std::abs(normed_value - (args->code[idx]));
            float dist_right = std::abs(normed_value - (args->code[idx + 1]));
            if (dist_right < dist_left) { idx += 1; }
        }

        // 5. store index
        args->out[i] = (unsigned char) idx;
    }

    return NULL;
}


// --------------------------------------

struct QTile
{
  float scaler{1.0f};
  float bmin{0.0f};
  float bmax{0.0f};
  float offset{0.0f};
};

#if defined(__AVX__)
// a: uint8, b: int8
void vpdpbusd_avi(__m256i &acc, __m256i a, __m256i b) {

  acc = _mm256_dpbusd_epi32(acc, a, b);
}
#endif

void vpdpbusd_C_16(int *dst, const uint8_t *u, const int8_t *s)
{
    for (int i = 0; i < 16; i++) {
        int sum = dst[i];
        for (int j = 0; j < 4; j++) {
            sum += u[i * 4 + j] * s[i * 4 + j];
        }
        dst[i] = sum;
    }
}

void vpdpbusd_C_8(int *dst, const uint8_t *u, const int8_t *s)
{
    for (int i = 0; i < 8; i++) {
        int sum = dst[i];
        for (int j = 0; j < 4; j++) {
            sum += u[i * 4 + j] * s[i * 4 + j];
        }
        dst[i] = sum;
    }
}

// TODO: Use posix_memalign for pre C++17
template<typename T>
static inline T *malloc_aligned(size_t nbytes, size_t align_bytes = alignof(T)) noexcept
{
#if defined(_MSC_VER) || defined(__MINGW32__)
  return reinterpret_cast<T*>(::_aligned_malloc(nbytes, align_bytes));
#else
  void* p;
  //
  return reinterpret_cast<T*>(::aligned_alloc(align_bytes, nbytes));
#endif  // defined(_MSC_VER) || defined(__MINGW32__)
}

static inline void
mfree_aligned(void* ptr) noexcept
{
#if defined(_MSC_VER) || defined(__MINGW32__)
  ::_aligned_free(ptr);
#else
  std::free(ptr);
#endif  // defined(_MSC_VER) || defined(__MINGW32__)
}


int main(int argc, char **argv)
{
  return 0;
}

