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
#ifndef _HF_IMAGE_DECODE_H
#define _HF_IMAGE_DECODE_H

#include "hf_codecs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************
 * 图片解码器核心流程接口
 **************************/
/**
 * @brief, 创建图片解码器实例
 *
 * @param cfg[in], 传入参数，创建图片解码器需要用到的参数，shf_image_dec_cfg请见hf_codecs_common.h中定义
 * @param inst[out], 传出参数，图片解码器实例，创建成功会返回可用的解码器实例，失败会返回NULL
 * @return 返回0为成功，小于0为失败
 * @note 该函数需与hf_destroy_image_decoder成对出现
 */
int hf_create_image_decoder(shf_image_dec_cfg cfg, HFIMAGE_DECODER* inst);

/**
 * @brief, 设置图片解码配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_image_decoder返回的实例inst
 * @param cfg[in]], 传入参数，图片解码使用的配置参数
 * @return 返回0为成功，小于0为失败
 * @note 该函数用于更新解码配置参数，会覆盖hf_create_image_decoder传入的参数
 */
int hf_set_image_decoder_config(HFIMAGE_DECODER inst, shf_image_dec_cfg cfg);

/**
 * @brief, 获取图片配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_image_decoder返回的实例inst
 * @param cfg[out], 传出参数，图片解码器当前使用的配置参数
 * @return 返回0为成功，小于0为失败
 * @note 该函数为hf_set_image_decoder_config提供先决条件
 */
int hf_get_image_decoder_config(HFIMAGE_DECODER inst, shf_image_dec_cfg* cfg);

/**
 * @brief, 解码一张图片
 *
 * @param inst[in], 传入参数，传入hf_create_image_decoder返回的实例inst
 * @param input[in], 传入参数，解码需要的数据及参数，参考hf_codecs_common.h中的simage_dec_input
 * @param output[out], 传出参数，解码后的数据及参数，用法与说明参考hf_codecs_common.h中的simage_dec_output,
 * 其中的output->buffer必须手动调用hf_freep()做释放，否则会有内存泄漏
 * @return 返回0为成功，小于0为失败
 * @note 无
 */
int hf_decode_image(HFIMAGE_DECODER inst, const simage_dec_input input, simage_dec_output* output);

/**
 * @brief, 销毁图片解码器实例
 *
 * @param inst[in], 传入参数，传入hf_create_image_decoder返回的实例inst
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需与hf_create_image_decoder成对出现
 */
int hf_destroy_image_decoder(HFIMAGE_DECODER inst);



/***************************
 * 图片解码器辅助接口
 **************************/
/**
 * @brief, 获取该平台下最佳图片解码后的输出颜色格式
 *
 * @param pixfmt[out], 传出参数，该平台下最佳图片解码后的输出颜色格式
 * @return 返回0为成功，小于0为失败
 * @note 用于预置相应平台执行效率最高的图片解码后颜色格式；不依赖图片解码器实例，可在任意时间使用
 */
int hf_image_dec_optimal_output_pixfmt(hf_pixfmt* pixfmt);

#ifdef __cplusplus
}
#endif


#endif