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
#ifndef _HF_VIDEO_DECODE_H
#define _HF_VIDEO_DECODE_H

#include "hf_codecs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************
 * 视频拉流解码器核心流程接口
 **************************/
/**
 * @brief, 创建视频解码器实例
 *
 * @param cfg[in], 输入参数，创建视频解码器需要用到的参数，shf_video_dec_cfg请见hf_codecs_common.h中定义
 * @param inst[out], 输出参数，视频解码器实例，创建成功会返回可用的解码器实例，失败会返回NULL
 * @return 返回0为成功，小于0为失败
 * @note 此函数只会创建解码器实例，不会与摄像机（或流媒体服务）建立连接，需与hf_destroy_video_decoder成对出现
 */
int hf_create_video_decoder(shf_video_dec_cfg cfg, HFVIDEO_DECODER *inst);

/**
 * @brief, 设置（更新）视频解码器参数
 *
 * @param inst[in], 输入参数，传入hf_create_video_decoder返回的实例inst
 * @param cfg[in], 输入参数，视频解码器工作需要用到的参数，shf_video_dec_cfg请见hf_codecs_common.h中定义
 * @return 返回0为成功，小于0为失败
 * @note 该函数主要用于更新配置，如果用在hf_create_video_decoder之后，hf_video_decoder_open之前，配置会立即生效，
 * 否则在下一次调hf_video_decoder_open时才会生效；若只有部分参数需要修改，可用hf_get_video_decoder_config先获取到
 * 当前配置，然后修改参数后再整个传入；此函数传入的配置会覆盖掉调hf_create_video_decoder函数时传入的配置
 */
int hf_set_video_decoder_config(HFVIDEO_DECODER inst, shf_video_dec_cfg cfg);

/**
 * @brief, 获取视频解码器参数
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @param cfg[out], 传出参数，视频解码器当前的配置参数，由调用方申请空间和释放，shf_video_dec_cfg请见
 * hf_codecs_common.h中定义
 * @return 返回0为成功，小于0为失败
 * @note 该函数为hf_set_video_decoder_config函数提供先决条件
 */
int hf_get_video_decoder_config(HFVIDEO_DECODER inst, shf_video_dec_cfg* cfg);

/**
 * @brief, 打开视频流（与摄像机、流媒体服务建立连接或打开文件）并打开其内部解码器上下文
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @param video_param[out], 传出参数，返回视频信息，shf_video_param具体请见hf_codecs_common.h中定义
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需与hf_video_stream_decoder_close成对出现
 */
int hf_video_stream_decoder_open(HFVIDEO_DECODER inst, shf_video_param* video_param);

/**
 * @brief, 关闭视频流（与摄像机、流媒体服务断开连接或关闭文件）并关闭其内部解码器上下文
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需与hf_video_stream_decoder_open成对出现
 */
int hf_video_stream_decoder_close(HFVIDEO_DECODER inst);

/**
 * @brief, 获取一帧解码后的图像数据
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @param out_buffer[in|out], 传入传出参数，解码后的原始图像线性数据，yuv、nv12等格式，由调用方事先分配好内存，
 * 需分配的大小使用hf_video_stream_decoder_open返回的video_param参数，传入hf_get_decoded_data_size获取
 * @param buffer_size[in], 传入参数，out_buffer的长度，单位为Byte
 * @param output[out], 传出参数，svideo_dec_output请见hf_codecs_common.h中定义
 * @return 返回0为成功，小于0为失败
 * @note 调用方需循环调用该函数以获取连续的视频帧数据，若返回失败，需触发重连（先hf_video_stream_decoder_close，
 * 再hf_video_stream_decoder_open）
 */
int hf_get_decoded_data(HFVIDEO_DECODER inst, unsigned char* out_buffer, const int buffer_size, svideo_dec_output* output);                        

/**
 * @brief, 销毁视频解码器实例
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @return 返回0为成功，小于0为失败
 * @note 该函数据需与hf_create_video_decoder成对出现；调用此函数前需先调用hf_video_stream_decoder_close，关闭掉
 * 视频流及内部解码器
 */
int hf_destroy_video_decoder(HFVIDEO_DECODER inst);



/***************************
 * 视频解码器偏业务侧接口
 **************************/
/**
 * @brief, 设置短视频录制配置参数
 *
 * @param inst[in], 输入参数，传入hf_create_video_decoder返回的实例inst
 * @param cfg[in], 输入参数，短视频录制配置参数，sflv_record_cfg请见hf_codecs_common.h中定义
 * @return 返回0为成功，小于0为失败
 * @note 该函数主要用于设置或更新短视频录制配置，不配置不会影响视频解码核心流程，也相当于不启动短视频录制功能；若
 * 要更新配置，需先做hf_get_flv_record_config以获取当前配置，只修改需要更新的字段即可
 */
int hf_set_flv_record_config(HFVIDEO_DECODER inst, sflv_record_cfg cfg);

/**
 * @brief, 获取设置短视频录制配置参数
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @param cfg[out], 传出参数，短视频录制配置参数，sflv_record_cfg请见hf_codecs_common.h中定义
 * @return 返回0为成功，小于0为失败
 * @note 该函数可做为hf_set_video_decoder_config函数的前置调用
 */
int hf_get_flv_record_config(HFVIDEO_DECODER inst, sflv_record_cfg* cfg);

/**
 * @brief, 短视频录制触发
 *
 * @param inst[in], 传入参数，传入hf_create_video_decoder返回的实例inst
 * @param alarm_pts[in], 传入参数，传入hf_get_decoded_data中获取到的frame_pts
 * @param detail_id[in], 传入参数，传入告警的detail_id
 * @param detail_id[in], 传入参数，传入短视频文件名称，若使用内部生成的名称，请传入空
 * @return 返回0为成功，小于0为失败
 * @note 短视频录制只需触发一下即可，是否录制成功及相关信息，会通过回调函数异步返回；此函数会立即返回
 */
int hf_trig_flv_record(HFVIDEO_DECODER inst, const int64_t alarm_pts, const unsigned char* detail_id, const unsigned char* file_name);



/***************************
 * 视频解码器辅助接口
 **************************/
/**
 * @brief, 获取该平台下最佳视频解码后的输出颜色格式
 *
 * @param pixfmt[out], 传出参数，该平台下最佳视频解码后的输出颜色格式
 * @return 返回0为成功，小于0为失败
 * @note 用于预置相应平台执行效率最高的视频解码后颜色格式；不依赖视频解码器实例，可在任意时间使用
 */
int hf_video_dec_optimal_output_pixfmt(hf_pixfmt* pixfmt);

#ifdef __cplusplus
}
#endif


#endif