#include <vector>

#include "caffe/layers/conv_approx_layer.hpp"

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
#define ITER 1
#define BITMASK ~((1<<6)-1)

namespace caffe {


__device__ void float2bfloat_conv(const float src, float& dst) {
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

__device__ uint8_t LOD_conv(uint8_t val){
    uint32_t n = 0, x;
    x = val;
    if (x <= 0x0000ffff) n += 16, x <<= 16;
    if (x <= 0x00ffffff) n += 8, x <<= 8;
    if (x <= 0x0fffffff) n += 4, x <<= 4;
    if (x <= 0x3fffffff) n += 2, x <<= 2;
    if (x <= 0x7fffffff) n++;
    return 31 - n;
}

__device__ uint16_t ILM_conv(uint8_t a, uint8_t b, uint8_t iter){
    /*
        a, b -> input operands,
        iter -> number of iterations
        only two iterations supported
    */
    if (a == 0 || b == 0) return 0;

    uint8_t Ka, Kb; 
    Ka = LOD_conv(a);
    Kb = LOD_conv(b);

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
        Ka = LOD_conv(ResA);
        Kb = LOD_conv(ResB);
        Res2B = ResB ^ (1 << Kb);
        prod1 = ResA * (1<<Kb) + Res2B * (1<<Ka);
    }

    return prod0 + (prod1);
}

__device__ uint32_t fp32_mul_core_conv (uint32_t a, uint32_t b, uint8_t iter)
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
    uint16_t p_short = ILM_conv(mantA_short,mantB_short,iter);

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

__device__  uint32_t uint_as_floatV2_conv (float a)
{
    uint32_t r;
    memcpy (&r, &a, sizeof r);
    return r;
}

__device__ float float_as_uintV2_conv (uint32_t a)
{
    float r;
    memcpy (&r, &a, sizeof r);
    return r;
}

__device__ float fp32_mul_ILM_conv (float a, float b, uint8_t iter)
{
    return float_as_uintV2_conv(fp32_mul_core_conv (uint_as_floatV2_conv(a), uint_as_floatV2_conv(b),iter));
}




  template <typename Dtype>
__global__ void ConvForward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w, const int kernel_n,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
    const int ph = (index / conved_width) % conved_height;
    // kernel_n denotes the number of filters which is equal to the number of channels 
    const int c = (index / conved_width / conved_height) % kernel_n;
		const int n = index / conved_width / conved_height / kernel_n;
    
    int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
//		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
    Dtype aveval = 0;
    
//		if (index==1) {
//			printf("pw%d ph%d c%d n%d \n",pw,ph,c,n);
//			printf("hstart%d wstart%d hend%d wend%d \n",hstart,wstart,hend,wend);
//		}
    for(int ch = 0; ch < channels; ++ch){
      int khstart=hend<kernel_h?kernel_h-hend:0;
      int kwstart=wend<kernel_w?kernel_w-wend:0;
      const Dtype*  bottom_slice = bottom_data + (n * channels + ch) * height * width;
      const Dtype*  weight_slice = weight + (c * channels + ch) * kernel_h * kernel_w;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {

          aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
          //aveval += mult_fixed_conv(&bottom_slice[h * width + w],&weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)]);

        }
      }
    }
		if(bias_term_) {  
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}

  template <typename Dtype>
__global__ void ConvForward_bfloat(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w, const int kernel_n,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
    const int ph = (index / conved_width) % conved_height;
    // kernel_n denotes the number of filters which is equal to the number of channels 
    const int c = (index / conved_width / conved_height) % kernel_n;
		const int n = index / conved_width / conved_height / kernel_n;
    
    int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
//		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
    Dtype aveval = 0;
    
//		if (index==1) {
//			printf("pw%d ph%d c%d n%d \n",pw,ph,c,n);
//			printf("hstart%d wstart%d hend%d wend%d \n",hstart,wstart,hend,wend);
//		}
    for(int ch = 0; ch < channels; ++ch){
      int khstart=hend<kernel_h?kernel_h-hend:0;
      int kwstart=wend<kernel_w?kernel_w-wend:0;
      const Dtype*  bottom_slice = bottom_data + (n * channels + ch) * height * width;
      const Dtype*  weight_slice = weight + (c * channels + ch) * kernel_h * kernel_w;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {

          float A = bottom_slice[h * width + w];
			    float B = weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
			    float tempA = 0;
			    float tempB = 0;
			    float2bfloat_conv(A, tempA);
			    float2bfloat_conv(B, tempB);
			    float mult = tempA * tempB;
			    float real_ma_out = 0;
            //printf("A: %4.4f, B: %4.4f, P: %4.4f\n",A,B,real_ma_out);
          float2bfloat_conv(mult,real_ma_out);
          aveval += real_ma_out;
          //aveval += mult_fixed_conv(&bottom_slice[h * width + w],&weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)]);

        }
      }
    }
		if(bias_term_) {  
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}

  template <typename Dtype>
__global__ void ConvForward_float(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w, const int kernel_n,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
    const int ph = (index / conved_width) % conved_height;
    // kernel_n denotes the number of filters which is equal to the number of channels 
    const int c = (index / conved_width / conved_height) % kernel_n;
		const int n = index / conved_width / conved_height / kernel_n;
    
    int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
//		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
    Dtype aveval = 0;
    
//		if (index==1) {
//			printf("pw%d ph%d c%d n%d \n",pw,ph,c,n);
//			printf("hstart%d wstart%d hend%d wend%d \n",hstart,wstart,hend,wend);
//		}
    for(int ch = 0; ch < channels; ++ch){
      int khstart=hend<kernel_h?kernel_h-hend:0;
      int kwstart=wend<kernel_w?kernel_w-wend:0;
      const Dtype*  bottom_slice = bottom_data + (n * channels + ch) * height * width;
      const Dtype*  weight_slice = weight + (c * channels + ch) * kernel_h * kernel_w;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {

          float A = bottom_slice[h * width + w];
			    float B = weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
			    float mult = A * B;
          aveval += mult;
        }
      }
    }
		if(bias_term_) {  
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}


  template <typename Dtype>
__global__ void ConvForward_ILM1(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w, const int kernel_n,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
    const int ph = (index / conved_width) % conved_height;
    // kernel_n denotes the number of filters which is equal to the number of channels 
    const int c = (index / conved_width / conved_height) % kernel_n;
		const int n = index / conved_width / conved_height / kernel_n;
    
    int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
//		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
    Dtype aveval = 0;
    
//		if (index==1) {
//			printf("pw%d ph%d c%d n%d \n",pw,ph,c,n);
//			printf("hstart%d wstart%d hend%d wend%d \n",hstart,wstart,hend,wend);
//		}
    for(int ch = 0; ch < channels; ++ch){
      int khstart=hend<kernel_h?kernel_h-hend:0;
      int kwstart=wend<kernel_w?kernel_w-wend:0;
      const Dtype*  bottom_slice = bottom_data + (n * channels + ch) * height * width;
      const Dtype*  weight_slice = weight + (c * channels + ch) * kernel_h * kernel_w;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          
          float A = bottom_slice[h * width + w];
			    float B = weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
			    float tempA = 0;
			    float tempB = 0;
			    float2bfloat_conv(A, tempA);
			    float2bfloat_conv(B, tempB);
			    float mult = fp32_mul_ILM_conv(tempA,tempB,ITER);
			    float real_ma_out = 0;
            //printf("A: %4.4f, B: %4.4f, P: %4.4f\n",A,B,real_ma_out);
          float2bfloat_conv(mult,real_ma_out);
          aveval += real_ma_out;
          //aveval += mult_fixed_conv(&bottom_slice[h * width + w],&weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)]);

          //aveval += mult_fixed_conv(&bottom_slice[h * width + w],&weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)]);

        }
      }
    }
		if(bias_term_) {  
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}


  // This code needs to be modified 
template <typename Dtype>
void ConvolutionApproxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //	std::cout << "fp" << std::endl;
	const Dtype* weight = this->blobs_[0]->gpu_data();
	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();

	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		const int count = top[i]->count();
		vector<int> shape_ = bottom[i]->shape();
		const int channels_ = shape_[1];
		const int height_ = shape_[2];
		const int width_ = shape_[3];

    
    // number_of_outputs 
    vector<int> weight_shape_ = top[i]->shape();
    const int kernel_n_ = weight_shape_[1];


    const int kernel_h_ = kernel_shape_data[0];
		const int kernel_w_ = kernel_shape_data[1];
		const int stride_h_ = stride_data[0];
		const int stride_w_ = stride_data[1];
		const int pad_h_ = pad_data[0];
		const int pad_w_ = pad_data[1];

		const int conved_height = this->output_shape_[0];
		const int conved_weight = this->output_shape_[1];
    
		const bool bias_term_ = this->bias_term_;

		if (bias_term_) {
			const Dtype* const bias = this->blobs_[1]->gpu_data();

      switch(mult_type){
        case 2: 
          ConvForward_bfloat<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
          break;
        case 3: 
          ConvForward_ILM1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
          break;
        case 4: 
          ConvForward_float<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
          break;
        default:
          printf("Wrong multiplier \n");
          return;
      }
/*		ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
 */ 
		} else {
		/*	ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
		*/
    switch(mult_type){
        case 2: 
          ConvForward_bfloat<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
		
        case 3: 
          ConvForward_ILM1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
          break;
        case 4: 
          ConvForward_float<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
        default:
          printf("Wrong multiplier \n");
          return;
      }
    
    
    }
	}
}


template <typename Dtype>
void ConvolutionApproxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_approx(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm_approx(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionApproxLayer);

}  // namespace caffe
