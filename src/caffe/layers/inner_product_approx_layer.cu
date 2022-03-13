

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_approx_layer.hpp"
#include "caffe/util/math_functions.hpp"


#define P 12
#define MAX1 1<<(15-P)

namespace caffe{

__device__   int leadingBitPosition_fc(int val)
  {
    unsigned n = 0, x;
    x = val;
    if (x <= 0x0000ffff) n += 16, x <<= 16;
    if (x <= 0x00ffffff) n += 8, x <<= 8;
    if (x <= 0x0fffffff) n += 4, x <<= 4;
    if (x <= 0x3fffffff) n += 2, x <<= 2;
    if (x <= 0x7fffffff) n++;
    return 31 - n;
  }

// logarithmic product approximation 
__device__ int LPPG_fc(int x, int y, char qx, char qy) {

  unsigned short x0_abs, y_abs;
  int  ilm_as, ilm_bs,x00;
  int ilm_s;
  char sgn_x = x >= 0 ? 0 : 1;
  char sgn_y = y >= 0 ? 0 : 1;
  //x0_abs = sgn_x ? -(x)  : x;
  //y_abs = sgn_y ? -(y) : y;
  x0_abs = sgn_x ? -x-1  : x;
  y_abs = sgn_y ? -y-1 : y;
  char k1_tmp = leadingBitPosition_fc(x0_abs);
  char k2_tmp = leadingBitPosition_fc(y_abs);
  unsigned int k1, sk2;
  // quantization
  k1 = k1_tmp >= qx ? k1_tmp : 0;
  sk2 = k2_tmp >= qy ? (1 << (k2_tmp- qy)) : 0;
  // substitution
  x00 = x0_abs - (1 << k1);

  // add these to the simulation
  ilm_bs = ((sgn_x ^ sgn_y) ? (-x00-1)*sk2 : x00*sk2)*((y != 0 ) && x0_abs != 0) ;
  ilm_as = ((sgn_x ^ sgn_y) ? (-y_abs-1)*(1 << k1) : y_abs * (1 << k1))*( x != 0 );


  ilm_s = (ilm_bs << qy) + (ilm_as);
  //printf(" ILM = %d,  ilm_bs = %d, ilm_as = %d \n", ilm_s, ilm_bs, ilm_as);
  //printf("  ILM_product: %d \t", ilm_s);

  return ilm_s;
}

__device__ int LOBO_fc(int x, int y, char d, char qx, char qy) {
  if(x == 0 | y == 0) return 0;  
	int sum = 0;
	int x0, x1, x0_signed,sum_lower;
	x1 = x >> d;
	//printf("\n x1 = %d", x1);
 	x0 = x % (1 << d);
	x0_signed = x0;
	int sd = 1 << d;
	if(x0 < -sd/2) x0_signed = x0 + sd;
	if(x0 >  sd/2) x0_signed = x0 - sd;
	// Caclulation of LSB 
	x1 += (x0_signed < 0);

	sum_lower = LPPG_fc(x0_signed,y,qx,qy);

	sum = (y * x1)*(1 << d) + sum_lower*(x0_signed != 0 && y != 0);

	return sum;
}


__device__ LM_fc( int a,  int b, unsigned short w) {
    unsigned short n;
	n = 16;
	if(a == 0 || b == 0) return 0;
	char sgn_a = a > 0 ? 0 : 1;
	char sgn_b = b > 0 ? 0 : 1;
	unsigned int a_abs = sgn_a ? -(a)-1  : a;
	unsigned int b_abs = sgn_b ? -(b)-1 : b;

	// mux 
	unsigned int a_sel = a_abs;
	unsigned int b_sel = b_abs;

	unsigned int k_a, x_a;
	k_a = leadingBitPosition_fc(a_sel);
	x_a = a_sel << (n - 1 - k_a);
    //printf("Xa = %x \n", x_a);
	unsigned int  k_b, x_b;
	k_b = leadingBitPosition_fc(b_sel);
	x_b = b_sel << (n - 1 - k_b);
    //printf("Xb = %x \n", x_b);

    unsigned int tmp, tmp_prim;
    tmp = (1<<(n-1))-1;
    tmp_prim = ((1<<(n-1)) - (1<<(n-w)));

	unsigned int y_a, y_b, tmp_a, tmp_b;
	tmp_a = x_a & tmp;
	y_a = x_a & tmp_prim;
	y_a = y_a | (1 << (n-w-1));
    //printf("Ya = %x \n", y_a);

	tmp_b = x_b & tmp;
	y_b = x_b & tmp_prim;
	y_b = y_b | (1 << (n-w-1));

	//printf("Yb = %x \n", y_b);
	//char tresh = Q;

	// We truncate mantissa 
	unsigned int y_l;

	y_l = (y_a + y_b) & tmp;
	// We set the LSB of k_a and k_b to zero 

	unsigned int k_l;

	k_l = k_a + k_b + (((y_a + y_b) & (tmp+1)) >> (n - 1));

	double m;
	unsigned int p_abs;
	m = (double)y_l / (1 << 15);

	p_abs = (unsigned int)((1 + m)*(1 << k_l));

	int zero = (a == 0) || (b == 0)  ;
	int p;
	p = (sgn_a ^ sgn_b)? -p_abs-1 : p_abs; 
	p = p*(1-zero);
	return p;
}


__device__ int ELM_fc(int x, int y, int w) {
	int sum = 0;
	int x0, x1, x0_signed, x0_abs;
	char sign;
	// X = X1*2^14 + X0
	// X1 = -2*x15+x14+x13
	// X0 = -x13*2^13 + sum_(i=0)^(12)(x_i*2^i)
	x1 = x >> 14;
 	x0 = x % (1 << 14);
	x0_signed = x0;
	if(x0 < -8192) x0_signed = x0 + 16384;
	if(x0 >  8192) x0_signed = x0 - 16384;
	// Caclulation of LSB 
	x0_abs = x0_signed;
	x1 += (x0_signed < 0);

	int y0, y1, y0_signed, y0_abs;
	// Y = Y1*2^14 + Y0
	// X1 = -2*y15+y14+y13
	// X0 = -y13*2^13 + sum_(i=0)^(12)(y_i*2^i)
	y1 = y >> 14;
 	y0 = y % (1 << 14);
	y0_signed = y0;
	if(y0 < -8192) y0_signed = y0 + 16384;
	if(y0 >  8192) y0_signed = y0 - 16384;
	// Caclulation of LSB 
	y0_abs = y0_signed;
	y1 += (y0_signed < 0);

	// Calculation of product 
	// PP_3 = X1*Y1

	// PP_2 = X1*Y0, PP_1 = Y1*X0

	int PP_1 = x1*y;
	if(PP_1 < 0){
		PP_1 = (PP_1 - 1) | 1; 
	}
	sum +=  PP_1 <<14;
	printf(" \t PP_1 = %d \n", PP_1);

	//int PP_0 = x0_signed*y0_signed;
	int PP_0 = LM_fc(x0_signed,y,w);
	printf("\t PP0 = %d \n",PP_0);
	
	sum += PP_0;

	return sum;

}

template <typename Dtype>
__device__ Dtype mult_fixed_fc(const Dtype *a, const Dtype *b)
{
  int x, y;
  int z;
  // Cutting off in quantization
  x = (short)(*a * (1 << P));
  y = (short)(*b * (1 << P));
  x = *a >= MAX1 ? (1<<15)-1 : x;
  x = *a <= -MAX1 ? -(1<<15) : x;
  y = *b >= MAX1 ? (1<<15)-1 : y;
  y = *b <= -MAX1 ? -(1<<15) : y;
  z = LOBO_fc(x,y,12,8,12); 
  return ((Dtype)z / (1 << 2 * P));
 //return *a * *b;
}

  template <typename Dtype>
__global__ void FCCForward(const int nthreads,
		const Dtype* bottom_data, const Dtype*  weight,
    Dtype* top_data, int M, int N, int K, const Dtype* bias,
    const int bias_term_, const Dtype* const bias_multiplier) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % N;
    const int ph = index / N;

    Dtype aveval = 0;
    
//		if (index==1) {
//			printf("pw%d ph%d c%d n%d \n",pw,ph,c,n);
//			printf("hstart%d wstart%d hend%d wend%d \n",hstart,wstart,hend,wend);
//		}


   
  
    for(int pk = 0; pk < K; pk++){

      // aveval += bottom_data[ph*K + pk]*weight[pk + pw*K];
      // aveval += mult_fixed((double)bottom_data[ph*K + pk],(double)weight[pk + pw*K]);
      aveval += mult_fixed_fc(bottom_data+ph*K + pk,weight + pk + pw*K);
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
    FCCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,bottom_data, weight, top_data, M_, N_, K_,bias,bias_term_,bias_multiplier_.gpu_data());
  } else {
    FCCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,bottom_data, weight, top_data, M_, N_, K_,0,bias_term_,bias_multiplier_.gpu_data());
  }
  //  printf("Print %d \n", bottom.size());

  // for (int i = 0; i < bottom.size(); ++i) {
  //   const Dtype* bottom_data = bottom[i]->gpu_data();
	// 	Dtype* top_data = top[i]->mutable_gpu_data();
	// 	const int count = top[i]->count();
  //   if (bias_term_) {
  //       const Dtype* const bias = this->blobs_[1]->gpu_data();
  //       FCCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //           count,bottom_data, weight, top_data, M_, N_, K_,bias,bias_term_);
  //     } else {
  //       FCCForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //           count,bottom_data, weight, top_data, M_, N_, K_,0,bias_term_);
  //     }
  // }


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
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductApproxLayer);

}  // namespace caffe