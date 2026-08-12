#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

void conv_gemm_avx_openmp(
    const float *unfolded,   // [P][K]
    const float *weight,     // [Cout][K]
    const float *bias,       // [Cout]
    float *output,           // [Cout][P]
    int P,
    int K,
    int Cout)
{
    /*
        Parallelize independent output elements.

        Every (oc, p) calculates one dot product.
    */
    #pragma omp parallel for collapse(2)
    for (int oc = 0; oc < Cout; oc++) {

        for (int p = 0; p < P; p++) {

            __m256 acc_vec = _mm256_setzero_ps();

            int k = 0;

            /*
                SIMD section.

                Process 8 FP32 values at once.
            */
            for (; k + 7 < K; k += 8) {

                __m256 input_vec =
                    _mm256_loadu_ps(
                        unfolded + p * K + k);

                __m256 weight_vec =
                    _mm256_loadu_ps(
                        weight + oc * K + k);

                acc_vec =
                    _mm256_fmadd_ps(
                        input_vec,
                        weight_vec,
                        acc_vec);
            }

            /*
                Horizontal reduction:

                acc_vec =
                [a0 a1 a2 a3 a4 a5 a6 a7]

                convert to scalar sum
            */
            float temp[8];

            _mm256_storeu_ps(temp, acc_vec);

            float sum =
                temp[0] +
                temp[1] +
                temp[2] +
                temp[3] +
                temp[4] +
                temp[5] +
                temp[6] +
                temp[7];

            /*
                Handle remaining K values
                when K is not divisible by 8.
            */
            for (; k < K; k++) {

                sum +=
                    unfolded[p * K + k] *
                    weight[oc * K + k];
            }

            /*
                Add Conv bias.
            */
            if (bias != NULL) {
                sum += bias[oc];
            }

            /*
                Output shape = [Cout][P]
            */
            output[oc * P + p] = sum;
        }
    }
}