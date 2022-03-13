#include <vector>

#include "caffe/layers/conv_approx_layer.hpp"
//#include "caffe/util/approx_mult.hpp"




#define P 12
#define MAX 1<<(15-P)

namespace caffe{

__device__  int leadingBitPosition_conv(int val)
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
__device__ int LPPG_conv(int x, int y, char qx, char qy) {

  unsigned short x0_abs, y_abs;
  int  ilm_as, ilm_bs,x00;
  int ilm_s;
  char sgn_x = x >= 0 ? 0 : 1;
  char sgn_y = y >= 0 ? 0 : 1;
  //x0_abs = sgn_x ? -(x)  : x;
  //y_abs = sgn_y ? -(y) : y;
  x0_abs = sgn_x ? -x-1  : x;
  y_abs = sgn_y ? -y-1 : y;
  char k1_tmp = leadingBitPosition_conv(x0_abs);
  char k2_tmp = leadingBitPosition_conv(y_abs);
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

__device__ int LOBO_conv(int x, int y, char d, char qx, char qy) {
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

	sum_lower = LPPG_conv(x0_signed,y,qx,qy);

	sum = (y * x1)*(1 << d) + sum_lower*(x0_signed != 0 && y != 0);

	return sum;
}


__device__ LM_conv( int a,  int b, unsigned short w) {
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
	k_a = leadingBitPosition_conv(a_sel);
	x_a = a_sel << (n - 1 - k_a);
    //printf("Xa = %x \n", x_a);
	unsigned int  k_b, x_b;
	k_b = leadingBitPosition_conv(b_sel);
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


__device__ int ELM_conv(int x, int y, int w) {
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
	int PP_0 = LM_conv(x0_signed,y,w);
	printf("\t PP0 = %d \n",PP_0);
	
	sum += PP_0;

	return sum;

}

template <typename Dtype>
__device__ Dtype mult_fixed_conv(const Dtype *a, const Dtype *b)
{
  int x, y;
  int z;
  // Cutting off in quantization
  x = (short)(*a * (1 << P));
  y = (short)(*b * (1 << P));
  x = *a >= MAX ? (1<<15)-1 : x;
  x = *a <= -MAX ? -(1<<15) : x;
  y = *b >= MAX ? (1<<15)-1 : y;
  y = *b <= -MAX ? -(1<<15) : y;
  z = LOBO_conv(x,y,12,8,12); 
  return ((Dtype)z / (1 << 2 * P));
 //return *a * *b;
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

          //aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
          aveval += mult_fixed_conv(&bottom_slice[h * width + w],&weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)]);

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
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
		} else {
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, kernel_n_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
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
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionApproxLayer);

}  // namespace caffe
