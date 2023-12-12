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
#ifndef _HF_SCALE_H
#define _HF_SCALE_H

#include "hf_codecs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief, 获取该平台下可支持的输入输出颜色格式列表
 *
 * @param src_pixfmts[in|out], 传入传出参数，长度为LITTLE_SMALL_LEN_BUFF的定长数组，返回该平台下缩放支持的输入颜色格式数组，第一个（下标为0）为最佳输入颜色格式
 * @param src_pixfmt_nums[in|out], 传入传出参数，src_pixfmts数组的有效长度
 * @param dst_pixfmts[in|out], 传入传出参数，长度为LITTLE_SMALL_LEN_BUFF的定长数组，返回该平台下缩放支持的输出颜色格式数组，第一个（下标为0）为最佳输出颜色格式
 * @param dst_pixfmt_nums[in|out], 传入传出参数，dst_pixfmts数组的有效长度
 * @return 返回0为成功，小于0为失败
 * @note 若此接口返回失败，则说明此平台下不支持该缩放接口。所以，此接口可做为该平台是否支持缩放的依据
 */
int hf_scale_support_pixfmts(hf_pixfmt src_pixfmts[LITTLE_SMALL_LEN_BUFF], int* src_pixfmt_nums,
                                   hf_pixfmt dst_pixfmts[LITTLE_SMALL_LEN_BUFF], int* dst_pixfmt_nums);

/**
 * @brief, 对图像进行缩放及颜色格式转换
 *
 * @param src_param[in], 传入参数，源图像参数，具体请参考simage_scale_param定义
 * @param dst_param[in|out], 传入传出参数，目标图像参数，具体请参考simage_scale_param定义
 * @return 返回0为成功，小于0为失败
 * @note 无
 */
int hf_scale(const simage_scale_param src_param, simage_scale_param* dst_param);

#ifdef __cplusplus
}
#endif

#endif
