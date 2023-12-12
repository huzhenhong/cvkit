/*************************************************************************
 * Copyright (C) [2022] by ZHENGJUE-AI, Inc.
 * zhengjue-ai hyper fast codec
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/
#ifndef _HF_IMAGE_ENCODE_H
#define _HF_IMAGE_ENCODE_H

#include "hf_codecs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************
 * 图片编码器核心流程接口
 **************************/
/**
 * @brief, 创建图片编码器实例
 *
 * @param cfg[in], 传入参数，创建图片编码器需要用到的参数，shf_image_enc_cfg请见hf_codecs_common.h中定义
 * @param inst[out], 传出参数，图片编码器实例，创建成功会返回可用的编码器实例，失败会返回NULL
 * @return 返回0为成功，小于0为失败
 * @note 该函数需与hf_destroy_image_encoder成对出现
 */
int hf_create_image_encoder(shf_image_enc_cfg cfg, HFIMAGE_ENCODER* inst);

/**
 * @brief, 设置图片编码配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_image_encoder返回的实例inst
 * @param cfg[in], 传入参数，图片编码器使用的配置参数
 * @return 返回0为成功，小于0为失败
 * @note 该函数用于更新编码配置参数，会覆盖hf_create_image_encoder传入的参数
 */
int hf_set_image_encoder_config(HFIMAGE_ENCODER inst, shf_image_enc_cfg cfg);

/**
 * @brief, 获取图片编码配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_image_encoder返回的实例inst
 * @param cfg[out], 传出参数，图片编码器当前使用的配置参数
 * @return 返回0为成功，小于0为失败
 * @note 该函数为hf_set_image_encoder_config提供先决条件
 */
int hf_get_image_encoder_config(HFIMAGE_ENCODER inst, shf_image_enc_cfg* cfg);

/**
 * @brief, 编码一张图片
 *
 * @param inst[in], 传入参数，传入hf_create_image_encoder返回的实例inst
 * @param input[in], 传入参数，输入的原始图像数据及参数，请参考hf_codecs_common.h中的
 * @param out_buffer[out], 传出参数，编码输出的图片二进制数据；每次调用内部都会申请内存，此数据需要在调用方使用结束后显式调用hf_freep()进行释放
 * @param out_buffer_len[out], 传出参数，out_buffer长度
 * @return 返回0为成功，小于0为失败
 * @note 请使用out_buffer_len做为out_buffer有效数据长度控制标尺
 */
int hf_encode_image(HFIMAGE_ENCODER inst, const simage_enc_input input, unsigned char** out_buffer, unsigned int* out_buffer_len);

/**
 * @brief, 销毁图片编码器实例
 *
 * @param inst[in], 传入参数，传入hf_create_image_encoder返回的实例inst
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需与hf_create_image_encoder成对出现
 */
int hf_destroy_image_encoder(HFIMAGE_ENCODER inst);



/***************************
 * 图片编码器辅助接口
 **************************/
/**
 * @brief, 获取该平台下最佳图像输入颜色格式
 *
 * @param pixfmt[out], 传出参数，该平台下最佳图像输入颜色格式
 * @return 返回0为成功，小于0为失败
 * @note 用于预置相应平台执行效率最高的图片编码输入颜色格式；不依赖图片编码器实例，可在任意时间使用
 */
int hf_image_enc_optimal_input_pixfmt(hf_pixfmt* pixfmt);

#ifdef __cplusplus
}
#endif


#endif