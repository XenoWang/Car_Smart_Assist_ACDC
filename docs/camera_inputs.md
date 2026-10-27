# 摄像头图像输入

当前管线、能见度数组评分/校准、天气属性预测和YOLO预测共用图像输入规则。输入先转换成RGB HWC uint8，再沿用各模型原有的缩放、归一化和阈值。

## 默认数组与显式帧

裸数组按HWC解释：二维是灰度，三通道是RGB，四通道是RGBA。uint8范围为0..255；float16/32/64默认范围为0..1。BGR、CHW、其他数值范围和原始YUV需要ImageFrame声明。转换不根据每帧min/max拉伸亮度。

```python
from car_smart_assist.data.image_input import ImageFrame, to_rgb_uint8

# OpenCV采集后给出的BGR数组
frame = ImageFrame(bgr_array, pixel_format="BGR")
result = pipeline.run(frame)

# 归一化RGB浮点数组可直接传入
result = pipeline.run(rgb_float_0_to_1)

# RGB浮点仍以0..255表示时必须声明，不能靠像素最大值猜测
frame = ImageFrame(rgb_float_0_to_255, value_range=(0, 255))

# 已经由ISP处理的12位RGB，以uint16容器保存，未与传感器RAW混用
frame = ImageFrame(processed_rgb12, value_range=(0, 4095))

# SDK给出CHW布局时显式声明
frame = ImageFrame(chw_rgb, pixel_format="RGB", layout="CHW")

# 所有当前模型的直接入口也接受相同的frame
scores = scorer.score_arrays([frame])
weather = weather_predictor.predict(frame)
detections = yolo_predictor.predict(frame)
```

uint16或其他实数整数容器须声明有效码值范围，不能从容器推断10/12/16位。整数范围须落在容器可表示范围内。显式上下界必须有限、递增；RGB值超出约定范围会报错。bool、complex、object和NaN/Inf不作为有效图像。

非8位固定映射到0..255后按floor(x+0.5)舍入。默认RGB uint8像素保持不变。RGBA/BGRA的alpha沿用弃置规则，不做自动合成或预乘alpha还原；非有限数据在弃置前报错。

默认to_rgb_uint8返回独立连续副本。copy=False供只读处理复用已规范的RGB uint8；若得到原数组别名，调用方须自行遵守只读约定。模型入口和管线不会改写调用方原图。

## YUV帧

支持线性YUYV/YUY2、UYVY、NV12、NV21、I420、YV12，以及P010/P012。每个YUV帧必须声明color_matrix为bt601或bt709，color_range为full或limited。这些值应来自相机/采集接口的实际配置。

```python
# NV12分离平面：Y为H×W；UV为H/2×W，或H/2×W/2×2
frame = ImageFrame(
    (y_plane, uv_plane), pixel_format="NV12",
    color_matrix="bt709", color_range="limited",
)
result = pipeline.run(frame)

# YUYV/UYVY的NumPy表示为H×W×2，或H×(2W)
frame = ImageFrame(
    yuyv_array, pixel_format="YUYV",
    color_matrix="bt601", color_range="limited",
)
```

8位YUV使用uint8，420宽高均须为偶数，422宽须为偶数。I420/YV12可传三个平面；NV12/NV21/P010/P012可传两个平面，或无歧义的合并二维数组。NumPy平面视图可带行填充；I420/YV12合并二维数组要求连续且无行填充，有填充时使用分离平面或显式原始缓冲。

P010按小端uint16中的高10位解释，低6位须为0；P012使用高12位，低4位须为0。通用层只解释线性布局，不把tiled表面当线性图像。色度按采样块复制恢复，按声明矩阵/量化范围转换，最终RGB裁剪到0..255。

## 原始缓冲

原始bytes与压缩单帧通过pixel_format区分：指定原始像素格式时，必须同时给出宽高、dtype及每平面的字节行跨度。dtype的多字节类型应明确字节序。SDK已经给出有效NumPy视图时，可以直接使用视图及其stride。

```python
frame = ImageFrame(
    raw_p010_bytes, pixel_format="P010", width=width, height=height,
    row_strides=(y_stride_bytes, uv_stride_bytes), dtype="<u2",
    color_matrix="bt709", color_range="limited",
)
rgb = to_rgb_uint8(frame)
```

缓冲长度、最小行跨度、平面尺寸、格式与dtype须一致。错误元数据会得到明确输入错误。GPU surface、DMABUF及具体相机SDK句柄先由相应采集层映射成主机缓冲或NumPy视图。

## 图片文件、PIL与压缩单帧

文件路径、普通PIL对象、JPEG/PNG压缩bytes均走同一入口；压缩bytes不指定pixel_format，颜色排列由解码器确定。

```python
result = pipeline.run("frame.jpg")
result = pipeline.run(jpeg_bytes)
result = pipeline.run(ImageFrame("processed12.png", value_range=(0, 4095)))
```

PNG16从原始路径或编码bytes保留位深后再转换。已被上游解码为8位的PIL RGB对象无法恢复原16位；需要保位深时传原始文件/bytes。有明确原PNG16证据的PIL对象会要求改用原始来源，防止误把8位解码结果当16位。

命令行可为显式图片指定同一固定码值范围：

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --image processed12.png --image-range 0 4095
```

benchmark_pipeline.py也提供--image-range；路径模式和预解码数组模式使用同一范围，参数随结果记录。现有RGB训练/评估文件、RGB缓存及BILINEAR/letterbox规则保持；采集新增高位深数据时，应携带正确码值范围转换后再入库。

## 采集边界与错误帧

传感器Bayer/RCCB/RCCC、位打包RAW10/RAW12、线性HDR及H.264/H.265视频流先经具体ISP/SDK/解码器处理。ImageFrame的整数范围只描述已经处理的像素码值，不代替白平衡、去马赛克或HDR色调映射。

pipeline.run_batch接收帧序列；单个ImageFrame或压缩bytes使用run。无效帧记录在skipped.input，维持输出等长同序，并按输入不可用处理；其他合法帧继续。不将无法解码的帧伪造为黑图。

格式适配完成后，实际车辆摄像头仍应以其输出元数据和实拍样本核对颜色、曝光及模型效果，并测量端到端延迟。
