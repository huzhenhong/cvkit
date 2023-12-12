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
#ifndef _HF_CODECS_API_H
#define _HF_CODECS_API_H

#include "hf_codecs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief, 获取libhfcodecs sdk版本号
 *
 * @param version[out]，传出参数，版本号字符串；字符串资源内部管理，无需外部做释放等操作 
 * @return 返回0为成功，小于0为失败
 * @note 在程序启动时调用一次即可，可打印到日志
 */
int hf_get_version(unsigned char** version);

/**
 * @brief, 创建libhfcodecs sdk的全局日志打印机
 *
 * @param loglevel[in]，传入参数，日志打印级别，调用时框架需根据hf_codecs_common.h中log_level的
 * 定义做一个日志级别转换
 * @param log_cb[in]，传入参数，日志回调函数指针，详细可见hf_codecs_common.h中的LOGCALLBACK
 * @return 返回0为成功，小于0为失败
 * @note 因本sdk可能会有多个不同框架调用，所以log_level不对某一框架做适配；该函数全局调用一次即可，
 * 若不调用或log_cb传入空值，会启动内部日志打印，仅会打印在标准输出设备；需与hf_destory_log_printer
 * 配对使用
 */
int hf_create_log_printer(log_level loglevel, LOGCALLBACK log_cb);

/**
 * @brief, 销毁libhfcodecs sdk的全局日志打印机
 *
 * @param 无
 * @return 返回0为成功，小于0为失败
 * @note 与hf_create_log_printer配对使用，全局调用一次
 */
int hf_destory_log_printer();

/**
 * @brief, 解析错误码
 *
 * @param code_id[in]，传入参数，各接口调用时返回的错误码
 * @param err_msg[out]，传出参数，该错误码对应的错误描述；字符串资源内部管理，无需外部做释放等操作
 * @return 返回0为成功，小于0为失败
 * @note 无
 */
int hf_parse_error(int code_id, unsigned char** err_msg);

/**
 * @brief, 计算原始图像占用的Byte数
 *
 * @param video_param[in]，传入参数，图像参数，详见hf_codecs_common.h中的shf_video_param
 * @return 小于0为失败，大于0为计算出来的长度
 * @note 此函数仅使用shf_video_param中的width、height和pixfmt三个字段，需保证此三个字段赋值正确
 */
int hf_get_decoded_data_size(shf_video_param video_param);

/**
 * @brief, 视频流在线状态探测
 *
 * @param stream_path[in]，传入参数，视频流地址
 * @return 返回0为成功（在线），小于0为不在线
 * @note 此函数执行时为同步阻塞模式，会在较长时间返回，建议在独立线程内执行
 */
int hf_stream_online_probe(const unsigned char* stream_path);

/**
 * @brief, 释放指针指向的内存，并将其值置为NULL
 *
 * @param ptr[in]，需要释放的指针的指针
 * @return 返回0为成功，小于0为失败
 * @note 此函数必须在显式指明要调用的地方才需调用，否则会引起崩溃或内存泄漏
 */
int hf_freep(void** ptr);

#ifdef __cplusplus
}
#endif
#endif