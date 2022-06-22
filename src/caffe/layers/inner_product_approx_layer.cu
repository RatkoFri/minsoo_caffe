#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_approx_layer.hpp"
#include "caffe/util/math_functions.hpp"

extern unsigned int mult_type;         // multiplier mode

#define FLOAT_MANT_BITS    (23)
#define FLOAT_EXPO_BITS    (8)
#define FLOAT_EXPO_BIAS    (127)
#define FLOAT_MANT_MASK    (~((~0u) << (FLOAT_MANT_BITS+1))) /* incl. integer bit */
#define EXPO_ADJUST        (1)   /* adjustment for performance reasons */
#define MIN_NORM_EXPO      (1)   /* minimum biased exponent of normals */
#define MAX_NORM_EXPO      (254) /* maximum biased exponent of normals */
#define INF_EXPO           (255) /* biased exponent of infinities */
#define EXPO_MASK          (~((~0u) << FLOAT_EXPO_BITS))
#define FLOAT_SIGN_MASK    (0x80000000u)
#define FLOAT_IMPLICIT_BIT (1 << FLOAT_MANT_BITS)
#define RND_BIT_SHIFT      (31)
#define RND_BIT_MASK       (1u << RND_BIT_SHIFT)
#define FLOAT_INFINITY     (0x7f800000)
#define FLOAT_INDEFINITE   (0xffc00000u)
#define MANT_LSB           (0x00000001)
#define FLOAT_QNAN_BIT     (0x00400000)
#define MAX_SHIFT          (FLOAT_MANT_BITS + 2)
#define ITER 2

namespace caffe {



__device__ void float2bfloat_fc(const float src, float& dst) {
	const uint16_t* p = reinterpret_cast<const uint16_t*>(&src);
	uint16_t* q = reinterpret_cast<uint16_t*>(&dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  q[0] = p[0];
  q[1] = 0;
#else
	q[0] = 0;
	q[1] = p[1];
#endif
}

__device__ uint8_t LOD_fc(uint8_t val){
    uint32_t n = 0, x;
    x = val;
    if (x <= 0x0000ffff) n += 16, x <<= 16;
    if (x <= 0x00ffffff) n += 8, x <<= 8;
    if (x <= 0x0fffffff) n += 4, x <<= 4;
    if (x <= 0x3fffffff) n += 2, x <<= 2;
    if (x <= 0x7fffffff) n++;
    return 31 - n;
}

__device__ uint16_t ILM_fc(uint8_t a, uint8_t b, uint8_t iter){
    /*
        a, b -> input operands,
        iter -> number of iterations
        only two iterations supported
    */
    if (a == 0 || b == 0) return 0;

    uint8_t Ka, Kb; 
    Ka = LOD_fc(a);
    Kb = LOD_fc(b);

    uint8_t ResA, ResB, Res2B;
    ResA = a ^ (1 << Ka);
    ResB = b ^ (1 << Kb);

    uint16_t prod0, prod1;
    prod0 = a * (1<<Kb) + ResB * (1<<Ka);
    prod1 = 0;
    if(iter == 2){
        if(ResA == 0 || ResB == 0) {
            return prod0;
        }
        Ka = LOD_fc(ResA);
        Kb = LOD_fc(ResB);
        Res2B = ResB ^ (1 << Kb);
        prod1 = ResA * (1<<Kb) + Res2B * (1<<Ka);
    }

    return prod0 + prod1;
}

__device__ uint32_t fp32_mul_core_fc (uint32_t a, uint32_t b, uint8_t iter)
{
    uint64_t prod;
    uint32_t expoa, expob, manta, mantb, shift;
    uint32_t r, signr, expor, mantr_hi, mantr_lo;

    /* split arguments into sign, exponent, significand */
    expoa = ((a >> FLOAT_MANT_BITS) & EXPO_MASK) - EXPO_ADJUST;
    expob = ((b >> FLOAT_MANT_BITS) & EXPO_MASK) - EXPO_ADJUST;
    manta = (a | FLOAT_IMPLICIT_BIT) & FLOAT_MANT_MASK;
    mantb = (b | FLOAT_IMPLICIT_BIT) & FLOAT_MANT_MASK;
    /* result sign bit: XOR sign argument signs */
    signr = (a ^ b) & FLOAT_SIGN_MASK;
    if ((expoa >= (MAX_NORM_EXPO - EXPO_ADJUST)) || /* at least one argument is special */
        (expob >= (MAX_NORM_EXPO - EXPO_ADJUST))) { 
        if ((a & ~FLOAT_SIGN_MASK) > FLOAT_INFINITY) { /* a is NaN */
            /* return quietened NaN */
            return a | FLOAT_QNAN_BIT;
        }
        if ((b & ~FLOAT_SIGN_MASK) > FLOAT_INFINITY) { /* b is NaN */
            /* return quietened NaN */
            return b | FLOAT_QNAN_BIT;
        }
        if ((a & ~FLOAT_SIGN_MASK) == 0) { /* a is zero */
            /* return NaN if b is infinity, else zero */
            return (expob != (INF_EXPO - EXPO_ADJUST)) ? signr : FLOAT_INDEFINITE;
        }
        if ((b & ~FLOAT_SIGN_MASK) == 0) { /* b is zero */
            /* return NaN if a is infinity, else zero */
            return (expoa != (INF_EXPO - EXPO_ADJUST)) ? signr : FLOAT_INDEFINITE;
        }
        if (((a & ~FLOAT_SIGN_MASK) == FLOAT_INFINITY) || /* a or b infinity */
            ((b & ~FLOAT_SIGN_MASK) == FLOAT_INFINITY)) {
            return signr | FLOAT_INFINITY;
        }
        if ((int32_t)expoa < (MIN_NORM_EXPO - EXPO_ADJUST)) { /* a is subnormal */
            /* normalize significand of a */
            manta = a & FLOAT_MANT_MASK;
            expoa++;
            do {
                manta = 2 * manta;
                expoa--;
            } while (manta < FLOAT_IMPLICIT_BIT);
        } else if ((int32_t)expob < (MIN_NORM_EXPO - EXPO_ADJUST)) { /* b is subnormal */
            /* normalize significand of b */
            mantb = b & FLOAT_MANT_MASK;
            expob++;
            do {
                mantb = 2 * mantb;
                expob--;
            } while (mantb < FLOAT_IMPLICIT_BIT);
        }
    }
    /* result exponent: add argument exponents and adjust for biasing */
    expor = expoa + expob - FLOAT_EXPO_BIAS + 2 * EXPO_ADJUST;
    mantb = mantb ; /* preshift to align result signficand */
    /* result significand: multiply argument signficands */
    uint8_t mantA_short = manta >> 16; // Take only 8 bits (1 plus 7 bits of mantissa)
    uint8_t mantB_short = mantb >> 16; // Take only 8 bits (1 plus 7 bits of mantissa)
    uint16_t p_short = ILM_fc(mantA_short,mantB_short,iter);

    prod = (uint64_t)p_short << 32;
    prod = prod << FLOAT_EXPO_BITS;
    mantr_hi = (uint32_t)(prod >> 32);
    mantr_lo = (uint32_t)(prod >>  0);
    /* normalize significand */
    if (mantr_hi < FLOAT_IMPLICIT_BIT) {
        mantr_hi = (mantr_hi << 1) | (mantr_lo >> (32 - 1));
        mantr_lo = (mantr_lo << 1);
        expor--;
    }
    if (expor <= (MAX_NORM_EXPO - EXPO_ADJUST)) { /* normal, may overflow to infinity during rounding */
        /* combine biased exponent, sign and signficand */
        r = (expor << FLOAT_MANT_BITS) + signr + mantr_hi;
        /* round result to nearest or even; overflow to infinity possible */
        r = r + ((mantr_lo == RND_BIT_MASK) ? (mantr_hi & MANT_LSB) : (mantr_lo >> RND_BIT_SHIFT));
    } else if ((int32_t)expor > (MAX_NORM_EXPO - EXPO_ADJUST)) { /* overflow */
        /* return infinity */
        r = signr | FLOAT_INFINITY;
    } else { /* underflow */
        /* return zero, normal, or smallest subnormal */
        shift = 0 - expor;
        if (shift > MAX_SHIFT) shift = MAX_SHIFT;
        /* denormalize significand */
        mantr_lo = mantr_hi << (32 - shift) | (mantr_lo ? 1 : 0);
        mantr_hi = mantr_hi >> shift;
        /* combine sign and signficand; biased exponent known to be zero */
        r = mantr_hi + signr;
        /* round result to nearest or even */
        r = r + ((mantr_lo == RND_BIT_MASK) ? (mantr_hi & MANT_LSB) : (mantr_lo >> RND_BIT_SHIFT));
    }
    return r;
}

__device__  uint32_t uint_as_floatV2_fc (float a)
{
    uint32_t r;
    memcpy (&r, &a, sizeof r);
    return r;
}

__device__ float float_as_uintV2_fc (uint32_t a)
{
    float r;
    memcpy (&r, &a, sizeof r);
    return r;
}

__device__ float fp32_mul_ILM_fc (float a, float b, uint8_t iter)
{
    return float_as_uintV2_fc(fp32_mul_core_fc (uint_as_floatV2_fc(a), uint_as_floatV2_fc(b),iter));
}


  template <typename Dtype>
__global__ void FCCForward_float(const int nthreads,
		const Dtype* bottom_data, const Dtype*  weight,
    Dtype* top_data, int M, int N, int K, const Dtype* bias,
    const int bias_term_, const Dtype* const bias_multiplier) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % N;
    const int ph = index / N;

    Dtype aveval = 0;
    
    for(int pk = 0; pk < K; pk++){

      aveval += bottom_data[ph*K + pk]*weight[pk + pw*K];
      
    }
     // Bias multiplier needs to be checked, I have a bad feeling that  something isn't working like it should. Still, we managed to 
     // create inner product. At the end filter were in shape of N*K not K*N
		 if(bias_term_) {  
		 	aveval+=bias[pw]*bias_multiplier[ph];
	  }
		top_data[index] = aveval;
	}
}

  template <typename Dtype>
__global__ void FCCForward_bfloat(const int nthreads,
		const Dtype* bottom_data, const Dtype*  weight,
    Dtype* top_data, int M, int N, int K, const Dtype* bias,
    const int bias_term_, const Dtype* const bias_multiplier) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % N;
    const int ph = index / N;

    Dtype aveval = 0;
    
    for(int pk = 0; pk < K; pk++){
      float A = bottom_data[ph*K + pk];
      float B = weight[pk + pw*K];
      float tempA = 0;
      float tempB = 0;
      float2bfloat_fc(A, tempA);
      float2bfloat_fc(B, tempB);
      float mult = tempA * tempB;
      float real_ma_out = 0;
        //printf("A: %4.4f, B: %4.4f, P: %4.4f\n",A,B,real_ma_out);
      float2bfloat_fc(mult,real_ma_out);
      aveval += real_ma_out;
      //aveval += bottom_data[ph*K + pk]*weight[pk + pw*K];
      
    }
     // Bias multiplier needs to be checked, I have a bad feeling that  something isn't working like it should. Still, we managed to 
     // create inner product. At the end filter were in shape of N*K not K*N
		 if(bias_term_) {  
		 	aveval+=bias[pw]*bias_multiplier[ph];
	  }
		top_data[index] = aveval;
	}
}

  template <typename Dtype>
__global__ void FCCForward_ILM1(const int nthreads,
		const Dtype* bottom_data, const Dtype*  weight,
    Dtype* top_data, int M, int N, int K, const Dtype* bias,
    const int bias_term_, const Dtype* const bias_multiplier) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % N;
    const int ph = index / N;

    Dtype aveval = 0;
    
    for(int pk = 0; pk < K; pk++){
      float A = bottom_data[ph*K + pk];
      float B = weight[pk + pw*K];
      float tempA = 0;
      float tempB = 0;
      float2bfloat_fc(A, tempA);
      float2bfloat_fc(B, tempB);
      float mult = fp32_mul_ILM_fc(tempA,tempB,ITER);
      float real_ma_out = 0;
        //printf("A: %4.4f, B: %4.4f, P: %4.4f\n",A,B,real_ma_out);
      float2bfloat_fc(mult,real_ma_out);
      aveval += real_ma_out;
      //aveval += bottom_data[ph*K + pk]*weight[pk + pw*K];
      
    }
     // Bias multiplier needs to be checked, I have a bad feeling that  something isn't working like it should. Still, we managed to 
     // create inner product. At the end filter were in shape of N*K not K*N
		 if(bias_term_) {  
		 	aveval+=bias[pw]*bias_multiplier[ph];
	  }
		top_data[index] = aveval;
	}
}


template <typename Dtype>
void InnerProductApproxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();

  if (bias_term_) {
    const Dtype* const bias = this->blobs_[1]->gpu_data();

    switch(mult_type){
        case 2: 
          FCCForward_bfloat<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count,bottom_data, weight, top_data, M_, N_, K_,bias,bias_term_,bias_multiplier_.gpu_data());
          break;
        case 3: 
          FCCForward_ILM1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count,bottom_data, weight, top_data, M_, N_, K_,bias,bias_term_,bias_multiplier_.gpu_data());
          break;
        case 4: 
          FCCForward_float<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count,bottom_data, weight, top_data, M_, N_, K_,bias,bias_term_,bias_multiplier_.gpu_data());
          break;
        default:
          printf("Wrong multiplier \n");
          return;
      }

    /*
    FCCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,bottom_data, weight, top_data, M_, N_, K_,bias,bias_term_,bias_multiplier_.gpu_data());
        */
  } else {

    switch(mult_type){
        case 2: 
          FCCForward_bfloat<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count,bottom_data, weight, top_data, M_, N_, K_,0,bias_term_,bias_multiplier_.gpu_data());
          break;
        case 3: 
          FCCForward_ILM1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count,bottom_data, weight, top_data, M_, N_, K_,0,bias_term_,bias_multiplier_.gpu_data());
          break;
        case 4: 
          FCCForward_float<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count,bottom_data, weight, top_data, M_, N_, K_,0,bias_term_,bias_multiplier_.gpu_data());
          break;
        default:
          printf("Wrong multiplier \n");
          return;
      }
    /*
    FCCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,bottom_data, weight, top_data, M_, N_, K_,0,bias_term_,bias_multiplier_.gpu_data());
        */
  }

}

template <typename Dtype>
void InnerProductApproxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm_approxV2<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm_approxV2<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv_approxV2<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm_approxV2<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm_approxV2<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductApproxLayer);

}  // namespace caffe