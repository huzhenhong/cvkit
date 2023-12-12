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
#ifndef _HF_CODECS_COMMON_H
#define _HF_CODECS_COMMON_H

#include <stdint.h>

/***************************
 * 公共定义
 **************************/
// 数组长度
#define LITTLE_SMALL_LEN_BUFF   8
#define SMALL_LEN_BUFF          64
#define MIDDLE_LEN_BUFF         256
#define BIG_LEN_BUFF            1024
#define LARGE_LEN_BUFF          2048

// 日志级别
typedef enum
{
    LogUNKNOWN  = 0,
    LogFATAL    = 1,
    LogERROR    = 2,
    LogWARN     = 3,
    LogINFO     = 4,
    LogDEBUG    = 5,
    LogTRACE    = 6,
    LogMAXNUM   = 7,
}log_level;

// 日志回调函数模板
// @param 日志字符串
typedef void (*LOGCALLBACK)(unsigned char*); 

// 颜色格式
typedef enum 
{
    HF_PIXFMT_NONE     =    -1,
    HF_PIXFMT_YUV420P  =    0,
    HF_PIXFMT_NV12     =    1,
    HF_PIXFMT_NV21,
    HF_PIXFMT_RGB24,
    HF_PIXFMT_BGR24,
    HF_PIXFMT_NB
}hf_pixfmt;

// 视频参数
// @note 声明结构体对象或指针后，请立即memset将存储空间都置0，以便使部分参数使用默认值
typedef struct _video_param_
{
    int width;                                          // 视频宽
    int height;                                         // 视频高
    double duration;                                    // 视频的长度，单位为秒。仅在视频文件时生效，实时流为0.0
    double fps;                                         // 视频的帧率
    hf_pixfmt pixfmt;                                   // 视频的颜色格式
}shf_video_param, *pshf_video_param;

// 返回值（错误码）正负转换，内部用来正转负，外部用来负转正
#define RETVALUE(X) (X*-1)

// 错误码
// 调用时，若直接使用此错误码做返回判断，可使用如下方式：
// int ret = hf_xxx(); if (HF_RET_STREAN_OFFLINE == RETVALUE(ret)) {}
typedef enum
{
    /* 通用错误码 */
    HF_RET_SUCCESS = 0,                                 // 成功或在线
    HF_RET_CREATE_FAILED = 1,                           // 创建失败
    HF_RET_GETCFG_FAILED,                               // 获取配置失败
    HF_RET_SETCFG_FAILED,                               // 配置失败
    HF_RET_INSTANCE_ISNULL,                             // 实例未创建
    HF_RET_DEVICE_CHANGE_NOT_ALLOWED,                   // 不允许修改设备ID
    /* 全局接口错误码 */
    HF_RET_STREAN_OFFLINE,                              // 视频流不在线
    HF_RET_GETIMGBUFSIZE_FAILED,                        // 获取图像缓存大小失败
    /* 拉流解码错误码 */
    HF_RET_STREAM_OPEN_FAILED,                          // 视频流打开失败
    HF_RET_VDECODE_RES_BUSY,                            // 视频解码器资源不足
    HF_RET_VDECODE_PARAM_ERROR,                         // 视频解码器传入参数错误
    HF_RET_STREAM_READ_DECODE_FAILED,                   // 视频读包或解码失败
    HF_RET_STREAM_READ_EOF,                             // 视频读取到结尾。若是视频文件，为读取结束，若是视频流，请尝试重连（close->open）
    HF_RET_TRIGFLVREC_FAILED,                           // 触发短视频录制失败
    /* 编码推流错误码 */
    HF_RET_VENCODE_OPEN_FAILED,                         // 视频编码器打开失败
    HF_RET_ADD_PUSHER_FAILED,                           // 添加推流地址失败
    HF_RET_DEL_PUSHER_FAILED,                           // 删除推流地址失败
    HF_RET_PUSHER_RECONNECTING,                         // 该推流地址正在重连
    HF_RET_VENCODE_FAILED,                              // 视频编码失败
    HF_RET_PUSH_FAILED,                                 // 推流失败
    HF_RET_VENCODE_CLOSE_FAILED,                        // 视频编码器关闭失败
    /* 图片解码编码错误码 */
    HF_RET_IDECODE_FAILED,                              // 图片解码失败
    HF_RET_IENCODE_FAILED,                              // 图片编码失败
    /* 缩放及颜色格式转换错误码 */
    HF_RET_NOFMT_IS_SUPPORT,                            // 没有可以支持的颜色格式
    HF_RET_SCALE_FAILED,                                // 缩放或颜色格式转换失败
    /* 其他 */
    HF_RET_ERRORCODE_INVALID,                           // 错误码无效
    HF_RET_NB,                                          // 错误码总个数
}hf_ret_ec;

/***************************
 * 视频解码器相关定义
 **************************/
// 视频解码实例
typedef void* HFVIDEO_DECODER;

/***************************
 * 短视频回调函数模板
 **************************/
/**
 * @brief, 短视频回调函数
 *
 * @param detail_id[out], 输出参数，告警detail_id
 * @param filepath[out], 输出参数，生成短视频路径
 * @param success[out], 输出参数，短视频是否生成成功，0为失败，1为成功
 * @param context[out], 输出参数，回调函数上下文，会将sflv_record_cfg::record_cb_context原样返回
 * @return 空
 * @note 可全局有一个函数实现体，也可以有多个函数实现体
 */
typedef void(*RECORDCALLBACK)(unsigned char* detail_id, unsigned char* filepath, int success, void* context);

// 解码模式
typedef enum{
    DM_FULLFRAMEDECODE = 0,                             // 全帧解码，默认值
    DM_IFRAMEDECODE                                     // 只解I帧
}decode_mode;

// 解码策略
typedef enum{
    DS_AUTO = 0,                                        // 自动选择最优解码器，默认值
    DS_CPUDECODE,                                       // 强制CPU解码
    DS_GPUDECODE                                        // 强制GPU（或硬件芯片）解码
}decode_strategy;

// 文件解码速度
typedef enum{
    FDS_FULLSPEED = 0,		                            // 全速，默认值
    FDS_1XSPEED,	                                    // 1倍速
    FDS_2XSPEED,		                                // 2倍速
    FDS_4XSPEED,		                                // 4倍速
    FDS_8XSPEED,		                                // 8倍速
    FDS_16XSPEED,		                                // 16倍速
}file_dec_speed;

// 解码配置参数
// @note 声明结构体对象或指针后，请立即memset将存储空间都置0，以便使部分参数使用默认值
typedef struct _video_dec_cfg_
{
    unsigned char stream_url[LARGE_LEN_BUFF];           // 视频流地址或者文件
    decode_mode dec_mode;                               // 解码模式
    file_dec_speed dec_speed;	                        // 文件解码速度
    hf_pixfmt out_pixfmt;                               // 解码输出颜色格式，可使用hf_video_dec_optimal_output_pixfmt获取的推荐值
    decode_strategy dec_strategy;                       // 解码策略
    int device_id;								        // 编码使用的gpu（编码芯片）id，起始值为0，目前仅在x86平台下有效
                                                        // 其他平台可以不填或者填任意值，默认为0

    unsigned char logtag[SMALL_LEN_BUFF];               // 日志打印时使用的标记，用于多实例时的日志区分
}shf_video_dec_cfg, *pshf_video_dec_cfg;

// 短视频录制配置
// @note 声明结构体对象或指针后，请立即memset将存储空间都置0，以便使部分参数使用默认值
typedef struct _flv_record_cfg_
{
    RECORDCALLBACK record_cb;                           // 短视频录制时用的回调函数
    void* record_cb_ctx;                                // 短视频录制时回调函数的上下文，会通过回调函数最后一个参数原样返回，可为NULL
    unsigned char record_path[BIG_LEN_BUFF];            // 短视频存储路径
    unsigned char yamdi_path[BIG_LEN_BUFF];             // yamdi路经
    int is_cache;                                       // 是否开启视频帧缓存，短视频若要开启，1为缓存，0为不缓存
}sflv_record_cfg, *psflv_record_cfg;

// 解码接口输出参数结构体
typedef struct _video_dec_output_
{
    int64_t frame_pts;                                  // 当前解码图像的时间戳，在短视频录制时需要透传
    double time_line;                                   // 当前解码图像的时间线，每次hf_video_decoder_open后，该值会从0.0开始，每帧递增，单位为秒
    unsigned int width;                                 // 当前视频帧的宽度
    unsigned int height;                                // 当前视频帧的高度
}svideo_dec_output, *psvideo_dec_output;


/***************************
 * 视频编码器相关定义
 **************************/
// 视频编码实例
typedef void* HFVIDEO_ENCODER;

// 编码输出的缩放模式
typedef enum 
{
    OSM_ORIGINOUTPUT = 0,		                        // 保持输入图像的原始宽高做视频编码
    OSM_SCALEOUTPUT				                        // 使用指定的宽高做视频编码，内部会对图像做缩放
}output_scale_mode;

// 视频编码策略
typedef enum
{
    ES_AUTO = 0,                                        // 自动选择最优编码器
    ES_CPUENCODE,                                       // 强制CPU编码
    ES_GPUENCODE                                        // 强制GPU（或硬件芯片）编码
}encode_strategy;

// 编码配置参数
// @note 除logtag字段外，其余字段均可在hf_video_encoder_pusher_open之前，使用hf_set_video_encoder_config进行更新
// @note 声明结构体对象或指针后，请立即memset将存储空间都置0，以便使部分参数使用默认值
typedef struct _video_enc_cfg_
{
    // 视频编码输出参数
    unsigned char oformat[SMALL_LEN_BUFF];              // 编码后视频输出的封装格式，如"flv"、"mp4"等，若使用rtmp推流，此处应传入"flv"
    double ofps;                                        // 编码后视频的帧率
    int32_t ovideo_width;                               // 编码后视频的宽度    
    int32_t ovideo_height;                              // 编码后视频的高度    
    hf_pixfmt opixfmt;                                  // 编码后视频帧的颜色格式
    int32_t ovideo_bitrates;                            // 编码后视频的码率，单位为bit，如输出2Mbps码率，此处应传入2*1024*1024=2097152
    output_scale_mode output_mode;                      // 编码缩放模式，详见output_scale_mode的说明
    unsigned char ovideo_codec_name[SMALL_LEN_BUFF];    // 编码器名称，如"h264"、"h265"
    int oauto_reconnect;                                // 是否自动断线重连，0为关闭，1为启动
    encode_strategy oenc_strategy;                      // 视频编码策略，详见encode_strategy的说明

    // 原始图像输入参数
    hf_pixfmt iimage_pixfmt;                            // 输入编码器图像的颜色格式，可使用hf_video_enc_optimal_input_pixfmt获取的推荐值
    int32_t iimage_width;                               // 输入编码器图像的宽
    int32_t iimage_height;                              // 输入编码器图像的高
    double ifps;										// 输入编码器图像的帧率，需为真实的输入帧率，若取值-1，则会取前三秒数据进行帧率自动探测

    // 其他参数
    int device_id;								        // 编码使用的gpu（编码芯片）id， 起始值为0，目前仅在x86平台下有效，
                                                        // 其他平台可以不填或者填任意值， 默认为0
    unsigned char logtag[SMALL_LEN_BUFF];               // 日志打印时使用的标记，用于多实例时的日志区分
}shf_video_enc_cfg, *pshf_video_enc_cfg;

// 编码器write接口返回结果结构体
typedef struct _enc_write_ret_ 
{
    unsigned char push_path[LARGE_LEN_BUFF];            // 推流地址
    int err_code;                                       // 错误码
}senc_write_ret, *psenc_write_ret;


/***************************
 * 图片解码器相关定义
 **************************/
// 图片解码实例
typedef void* HFIMAGE_DECODER;

// 图片解码配置参数
typedef struct _image_dec_cfg_
{
    int device_id;								        // 图片解码使用的gpu（解码芯片）id，起始值为0，目前仅在x86平台下有效，
                                                        // 其他平台可以不填或者填任意值，默认为0
    unsigned char logtag[SMALL_LEN_BUFF];               // 日志打印时使用的标记，用于多实例时的日志区分
}shf_image_dec_cfg, *pshf_image_dec_cfg;

// 图片解码输入参数
// @note 声明结构体对象或指针后，请立即memset将存储空间都置0，以便使部分参数使用默认值
typedef struct _image_dec_input_
{
    unsigned char* buffer;                              // 解码前图片的二进制数据
    unsigned int buffer_len;                            // buffer的长度，单位为Byte
    unsigned char type[SMALL_LEN_BUFF];                 // 图片类型，如“jpg、png”等
}simage_dec_input, *pimage_dec_input;

// 图片解码输出参数
typedef struct _image_dec_output_
{
    unsigned char* buffer;                              // 解码后的图像数据，线性结构；每次调用内部都会申请内存，此数据需要在调用方使用结束后显式调用hf_freep()进行释放
    unsigned int buffer_len;                            // out_buffer的长度，单位为Byte
    hf_pixfmt pixfmt;                                   // 解码后的图像颜色格式，可参考hf_image_dec_optimal_output_pixfmt获取的推荐值
    unsigned int width;                                 // 解码后的图像的宽度
    unsigned int height;                                // 解码后的图像的高度
}simage_dec_output, *pimage_dec_output;


/***************************
 * 图片编码器相关定义
 **************************/
// 图片编码实例
typedef void* HFIMAGE_ENCODER;

// 图片编码配置参数
typedef struct _image_enc_cfg_
{
    int device_id;								        // 图片编码使用的gpu（编码芯片）id，起始值为0，目前仅在x86平台下有效，
                                                        // 其他平台可以不填或者填任意值，默认为0
    unsigned char logtag[SMALL_LEN_BUFF];               // 日志打印时使用的标记，用于多实例时的日志区分
}shf_image_enc_cfg, *pshf_image_enc_cfg;

// 图片编码输入参数
// @note 声明结构体对象或指针后，请立即memset将存储空间都置0，以便使部分参数使用默认值
typedef struct _image_enc_input_
{
    unsigned char* buffer;                              // 原始图像线性数据，如yuv420p、nv12等格式数据
    unsigned int buffer_len;                            // buffer长度，单位为Byte
    hf_pixfmt pixfmt;                                   // buffer数据的颜色格式，可参考hf_image_enc_optimal_input_pixfmt获取的推荐值
    unsigned int in_width;                              // 原始图像的宽度
    unsigned int in_height;                             // 原始图像的高度
    unsigned int out_width;                             // 需要编码输出的图像宽度，若无需缩放，可与in_width相同
    unsigned int out_height;                            // 需要编码输出的图像高度，若无需缩放，可与in_height相同
}simage_enc_input, *pimage_enc_input;


/***************************
 * 原始图像缩放与颜色格式转换相关定义
 **************************/
// 缩放图像参数
typedef struct _image_scale_param_
{
    unsigned int width;                                 // 图像的宽度，输入输出均由调用方赋值
    unsigned int height;                                // 图像的高度，输入输出均由调用方赋值

    hf_pixfmt pixfmt;                                   // 图像的颜色格式，可从hf_image_scale_support_pixfmts获取最佳输入输出颜色格式，
                                                        // 输入输出均由调用方赋值

    unsigned char* buffer;                              // 1、输入时，此参数为图像线性结构的二进制数据，内存由调用方管理
                                                        // 2、输出时，此参数会填充图像线性结构的二进制数据，内存空间由调用方申请和管理，
                                                        // 需要的内存长度可使用hf_get_decoded_data_size接口获取
}simage_scale_param, *pimage_scale_param;

#endif