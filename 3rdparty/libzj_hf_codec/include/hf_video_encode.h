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
#ifndef _HF_VIDEO_ENCODE_H
#define _HF_VIDEO_ENCODE_H

#include "hf_codecs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************
 * 视频推流编码器核心流程接口
 **************************/
/**
 * @brief, 创建视频解码器实例
 *
 * @param cfg[in], 传入参数，创建视频编码器需要用到的参数，shf_video_enc_cfg请见hf_codecs_common.h中定义
 * @param inst[out], 传出参数，视频编码器实例，创建成功会返回可用的编码器实例，失败会返回NULL
 * @return 返回0为成功，小于0为失败
 * @note 该函数需与hf_destroy_video_encoder成对出现
 */
int hf_create_video_encoder(shf_video_enc_cfg cfg, HFVIDEO_ENCODER* inst);

/**
 * @brief, 设置编码器配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_video_encoder返回的实例inst
 * @param cfg[in], 传入参数，视频编码器使用的配置参数
 * @return 返回0为成功，小于0为失败
 * @note 该函数主要用于更新配置，生效时机在hf_video_encoder_pusher_close把所有推流编码器都关闭成功再调用
 * hf_video_encoder_pusher_open之前，或在第一次调用hf_video_encoder_pusher_open之前；若只有部分参数需要
 * 修改，可用hf_get_video_encoder_config先获取到当前配置，然后修改参数后再整个传入；此函数传入的配置会覆盖
 * 掉调hf_create_video_encoder函数时传入的配置
 */
int hf_set_video_encoder_config(HFVIDEO_ENCODER inst, shf_video_enc_cfg cfg);

/**
 * @brief, 获取编码器配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_video_encoder返回的实例inst
 * @param cfg[out], 传出参数，当前视频编码器配置，由调用方申请空间和释放，内部只做结构体成员赋值
 * @return 返回0为成功，小于0为失败
 * @note 该函数为hf_set_video_encoder_config函数提供先决条件
 */
int hf_get_video_encoder_config(HFVIDEO_ENCODER inst, shf_video_enc_cfg* cfg);

/**
 * @brief, 打开推流地址并打开内部使用的编码器上下文
 *
 * @param inst[in], 传入参数，传入hf_create_video_encoder返回的实例inst
 * @param push_path[in], 传入参数，推流地址
 * @return 返回0为成功，小于0为失败
 * @note 不同push_path可以多次调用该函数，内部会打开对应的推流器并自动判断内部使用编码器上下文的打开与否；
 * 同一个push_path如果已经打开且在正常工作则返回成功
 */
int hf_video_encoder_pusher_open(HFVIDEO_ENCODER inst, const unsigned char* push_path);

/**
 * @brief, 关闭推流地址并关闭内部使用的编码器上下文
 *
 * @param inst[in], 传入参数，传入hf_create_video_encoder返回的实例inst
 * @param push_path[in], 传入参数，传入需要关闭的推流地址；若此地址是唯一（最后）一路推流地址，则会同时关闭编码器；
 * 若需一次关闭当前所有推流地址及编码器，传入空值即可
 * @return 返回0为成功，小于0为失败
 * @note 该函数支持不同推流地址多次调用，同一推流地址重复调用时若已经关闭则会返回成功
 */
int hf_video_encoder_pusher_close(HFVIDEO_ENCODER inst, const unsigned char* push_path);

/**
 * @brief, 写入需编码的数据
 *
 * @param inst[in], 传入参数，传入hf_create_video_encoder返回的实例inst
 * @param frame_data[in], 传入参数，待编码的线性图像数据，如yuv、nv12等格式数据
 * @param frame_data_len[in], 传入参数，待编码的图像数据的长度
 * @param ret_info[out], 传出参数，返回的结构体指针数组，senc_write_ret定义请见hf_codecs_common.h中定义，内部申请内存空间，
 * 调用后需使用hf_freep()显式释放
 * @param ret_nums[out], 传出参数，ret_info数组的个数
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需循环调用
 */
int hf_write_image_data(HFVIDEO_ENCODER inst, const unsigned char* frame_data, const unsigned int frame_data_len,
                        senc_write_ret** ret_info, int* ret_nums);

/**
 * @brief, 销毁编码器实例
 *
 * @param inst[in], 传入参数，传入hf_create_video_encoder返回的实例inst
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需与hf_create_video_encoder成对出现，且需在hf_video_encoder_pusher_close把所有推流器及编码器关闭掉之后调用
 */
int hf_destroy_video_encoder(HFVIDEO_ENCODER inst);



/***************************
 * 视频编码器辅助接口
 **************************/
/**
 * @brief, 获取该平台下最佳视频编码器的输入颜色格式
 *
 * @param pixfmt[out], 传出参数，该平台下最佳视频编码器的输入颜色格式
 * @return 返回0为成功，小于0为失败
 * @note 用于预置相应平台执行效率最高的视频编码输入颜色格式；不依赖视频编码器实例，可在任意时间使用
 */
int hf_video_enc_optimal_input_pixfmt(hf_pixfmt* pixfmt);

#ifdef __cplusplus
}
#endif


#endif