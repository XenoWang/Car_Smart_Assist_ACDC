"""Normalize declared host-memory image frames without guessing camera encodings.

RGB inputs use fixed code-value ranges. YUV inputs require an explicit matrix
and quantization range; this module does not perform sensor RAW processing,
tone mapping, video decoding, or camera acquisition.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, TypeAlias, cast, overload

import numpy as np
from PIL import Image

ImageBuffer: TypeAlias = bytes | bytearray | memoryview
ImagePlane: TypeAlias = np.ndarray | ImageBuffer
ImageData: TypeAlias = np.ndarray | Image.Image | Path | str | ImageBuffer | tuple[ImagePlane, ...]


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """An image plus the metadata needed to interpret its pixels.

    Bare arrays default to RGB/RGBA/gray in HWC/HW order. Raw buffers require
    pixel_format, width, height, dtype and one byte row stride per plane.
    CHW RGB buffers use one shared row stride and H consecutive rows per
    channel. YUV planes always use their native HWC/HW storage order.
    """

    data: ImageData
    pixel_format: str | None = None
    layout: str = "HWC"
    value_range: tuple[float, float] | None = None
    color_matrix: str | None = None
    color_range: str | None = None
    width: int | None = None
    height: int | None = None
    row_strides: tuple[int, ...] | None = None
    dtype: str | np.dtype | type | None = None


ImageInput: TypeAlias = ImageData | ImageFrame

_RGB_FORMATS = {"RGB": 3, "BGR": 3, "RGBA": 4, "BGRA": 4, "GRAY": 1}
_YUV_FORMATS = {"YUYV", "UYVY", "NV12", "NV21", "I420", "YV12", "P010", "P012"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _pixel_format(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("pixel_format must be a string or None")
    value = value.upper()
    if value == "YUY2":
        value = "YUYV"
    if value not in _RGB_FORMATS and value not in _YUV_FORMATS:
        raise ValueError(
            f"Unsupported pixel_format {value!r}; use an ISP/SDK or decoder to produce "
            "RGB/BGR/RGBA/BGRA/GRAY or a supported linear YUV frame first"
        )
    return value


def _layout(frame: ImageFrame) -> str:
    if not isinstance(frame.layout, str):
        raise TypeError("layout must be 'HWC' or 'CHW'")
    layout = frame.layout.upper()
    if layout not in {"HWC", "CHW"}:
        raise ValueError("layout must be 'HWC' or 'CHW'")
    return layout


@overload
def _dimension(value: int | None, name: str, *, required: Literal[True]) -> int: ...


@overload
def _dimension(value: int | None, name: str, *, required: Literal[False] = False) -> int | None: ...


def _dimension(value: int | None, name: str, *, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise ValueError(f"Raw image buffers require explicit {name}")
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_dimensions(frame: ImageFrame, height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("图像必须具有非空高和宽 (height and width must be positive)")
    for name, actual in (("width", width), ("height", height)):
        declared = _dimension(getattr(frame, name), name)
        if declared is not None and declared != actual:
            raise ValueError(f"Declared {name}={declared} does not match image {name}={actual}")


@overload
def _declared_dtype(frame: ImageFrame, *, required: Literal[True]) -> np.dtype: ...


@overload
def _declared_dtype(frame: ImageFrame, *, required: Literal[False] = False) -> np.dtype | None: ...


def _declared_dtype(frame: ImageFrame, *, required: bool = False) -> np.dtype | None:
    if frame.dtype is None:
        if required:
            raise ValueError("Raw image buffers require explicit dtype, including byte order")
        return None
    try:
        dtype = np.dtype(frame.dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid image dtype declaration: {frame.dtype!r}") from exc
    if dtype.kind not in "uif" or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError("Image dtype must be a real integer or floating-point scalar dtype")
    return dtype


def _check_dtype(frame: ImageFrame, array: np.ndarray) -> None:
    declared = _declared_dtype(frame)
    if declared is not None and declared != array.dtype:
        raise ValueError(f"Declared dtype {declared} does not match image dtype {array.dtype}")


@overload
def _strides(frame: ImageFrame, count: int, *, raw: Literal[True]) -> tuple[int, ...]: ...


@overload
def _strides(
    frame: ImageFrame, count: int, *, raw: Literal[False] = False
) -> tuple[int, ...] | None: ...


def _strides(frame: ImageFrame, count: int, *, raw: bool = False) -> tuple[int, ...] | None:
    if frame.row_strides is None:
        if raw:
            raise ValueError("Raw image buffers require explicit row_strides in bytes per plane")
        return None
    if not isinstance(frame.row_strides, (tuple, list)) or len(frame.row_strides) != count:
        raise ValueError(f"row_strides must contain {count} byte row stride(s), one per plane")
    result = []
    for value in frame.row_strides:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("row_strides entries must be integers in bytes")
        if value == 0 or (raw and value < 0):
            raise ValueError("Raw row_strides must be positive; array row_strides must be nonzero")
        result.append(int(value))
    return tuple(result)


def _check_plane_metadata(frame: ImageFrame, planes: tuple[np.ndarray, ...]) -> None:
    declared = _strides(frame, len(planes))
    for index, plane in enumerate(planes):
        _check_dtype(frame, plane)
        if declared is not None and declared[index] != plane.strides[0]:
            raise ValueError(
                f"Declared row_strides[{index}]={declared[index]} does not match "
                f"array row stride {plane.strides[0]}"
            )


def _value_range(frame: ImageFrame) -> tuple[float, float] | None:
    values = frame.value_range
    if values is None:
        return None
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError("value_range must be a finite (lower, upper) pair")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise TypeError("value_range bounds must be real numbers")
    try:
        lower, upper = (float(v) for v in values)
    except OverflowError as exc:
        raise ValueError("value_range bounds must be finite") from exc
    integer_bounds = all(isinstance(v, Integral) for v in values)
    span = int(values[1]) - int(values[0]) if integer_bounds else upper - lower
    if not math.isfinite(lower) or not math.isfinite(upper) or span <= 0:
        raise ValueError("value_range must have finite bounds with lower < upper")
    if not math.isfinite(span):
        raise ValueError("value_range span must be finite")
    return lower, upper


def _rgb_uint8(array: np.ndarray, frame: ImageFrame, pixel_format: str | None) -> np.ndarray:
    layout = _layout(frame)
    if array.ndim not in {2, 3}:
        raise ValueError("Image arrays must have HW, HWC or explicitly declared CHW dimensions")
    if array.ndim == 3:
        if layout == "CHW":
            array = array.transpose(1, 2, 0)
        channels = array.shape[2]
    else:
        channels = 1
    height, width = array.shape[:2]
    _check_dimensions(frame, height, width)
    _check_plane_metadata(frame, (array,))
    if pixel_format is None:
        pixel_format = {1: "GRAY", 3: "RGB", 4: "RGBA"}.get(channels)
        if pixel_format is None:
            raise ValueError("图像通道数必须为 1、3 或 4 (GRAY, RGB or RGBA)")
    elif channels != _RGB_FORMATS[pixel_format]:
        raise ValueError(
            f"{pixel_format} 图像通道数必须为 {_RGB_FORMATS[pixel_format]}，实际为 {channels}"
        )
    dtype = array.dtype
    if dtype.kind not in "uif" or (dtype.kind == "f" and dtype.itemsize not in {2, 4, 8}):
        raise TypeError("Image pixels must be real integers or float16/float32/float64; bool is unsupported")
    if dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("图像含 NaN 或 Inf 非有限像素值，包括 alpha 通道")
    bounds = _value_range(frame)
    if bounds is None:
        if dtype == np.dtype(np.uint8):
            bounds = (0.0, 255.0)
        elif dtype.kind == "f":
            bounds = (0.0, 1.0)
        else:
            raise ValueError(f"Image dtype {dtype} requires an explicit value_range")
    lower, upper = bounds
    integer_codes = None
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        original_bounds = frame.value_range if frame.value_range is not None else bounds
        # Python scalar comparisons preserve int64/uint64 endpoints that a
        # float64 comparison would round outside the container or input range.
        original_lower, original_upper = (
            value.item() if isinstance(value, np.generic) else value for value in original_bounds
        )
        if original_lower < int(limits.min) or original_upper > int(limits.max):
            raise ValueError(f"value_range ({lower}, {upper}) exceeds dtype {dtype} container limits")
        integer_codes = (math.ceil(original_lower), math.floor(original_upper))
    canonical_uint8 = dtype == np.dtype(np.uint8) and bounds == (0.0, 255.0)
    check_lower, check_upper = integer_codes if integer_codes is not None else bounds
    if not canonical_uint8 and (np.any(array < check_lower) or np.any(array > check_upper)):
        raise ValueError(f"Image pixel values are outside declared value_range ({lower}, {upper})")
    # Check every channel before dropping alpha, and preserve default uint8 exactly.
    if not canonical_uint8:
        if integer_codes is not None and frame.value_range is not None and all(
            isinstance(v, Integral) for v in frame.value_range
        ):
            # Unsigned modular subtraction gives an exact nonnegative integer
            # difference even for a signed range crossing zero, before float
            # quantization; close int64/uint64 ranges retain their small deltas.
            code_lower, code_upper = (int(v) for v in frame.value_range)
            unsigned = np.dtype(f"u{dtype.itemsize}")
            origin = np.array(code_lower % (1 << (dtype.itemsize * 8)), dtype=unsigned)
            delta = np.subtract(array.astype(unsigned), origin, dtype=unsigned)
            mapped = delta.astype(np.float64) / float(code_upper - code_lower) * 255.0
        else:
            mapped = (array.astype(np.float64) - lower) / (upper - lower) * 255.0
        array = np.floor(np.clip(mapped, 0.0, 255.0) + 0.5).astype(np.uint8)
    if pixel_format == "GRAY":
        gray = array if array.ndim == 2 else array[..., 0]
        return np.repeat(gray[..., None], 3, axis=2)
    if pixel_format in {"BGR", "BGRA"}:
        return array[..., 2::-1]
    return array[..., :3]


def _buffer(data: ImageBuffer) -> memoryview:
    view = memoryview(data)
    if not view.c_contiguous:
        raise ValueError("Raw/encoded byte buffers must be C-contiguous; supply an ndarray view instead")
    return view.cast("B")


def _raw_plane(
    buffer: memoryview, offset: int, rows: int, columns: int, stride: int, dtype: np.dtype
) -> np.ndarray:
    row_bytes = columns * dtype.itemsize
    if stride < row_bytes or stride % dtype.itemsize:
        raise ValueError(
            f"Raw plane row stride {stride} must be at least {row_bytes} bytes and aligned to dtype {dtype}"
        )
    required = offset + (rows - 1) * stride + row_bytes
    if required > buffer.nbytes:
        raise ValueError(f"Raw buffer is truncated: need at least {required} bytes, got {buffer.nbytes}")
    return np.ndarray((rows, columns), dtype=dtype, buffer=buffer, offset=offset,
                      strides=(stride, dtype.itemsize))


def _raw_rgb(frame: ImageFrame, pixel_format: str) -> np.ndarray:
    if not isinstance(frame.data, (bytes, bytearray, memoryview)):
        raise TypeError("Raw RGB data must be a byte buffer")
    width = _dimension(frame.width, "width", required=True)
    height = _dimension(frame.height, "height", required=True)
    dtype = _declared_dtype(frame, required=True)
    strides = _strides(frame, 1, raw=True)
    buffer = _buffer(frame.data)
    channels = _RGB_FORMATS[pixel_format]
    if _layout(frame) == "CHW" and channels > 1:
        _raw_plane(buffer, (channels - 1) * height * strides[0], height, width, strides[0], dtype)
        return np.ndarray((channels, height, width), dtype=dtype, buffer=buffer,
                          strides=(height * strides[0], strides[0], dtype.itemsize))
    packed = _raw_plane(buffer, 0, height, width * channels, strides[0], dtype)
    if channels == 1:
        return packed
    # reshape may discard a singleton row's declared padding stride.
    return np.ndarray((height, width, channels), dtype=dtype, buffer=buffer,
                      strides=(strides[0], channels * dtype.itemsize, dtype.itemsize))


def _check_yuv_dtype(dtype: np.dtype, pixel_format: str) -> None:
    if pixel_format in {"P010", "P012"}:
        little_endian = dtype.byteorder == "<" or (dtype.byteorder == "=" and np.little_endian)
        if dtype.kind != "u" or dtype.itemsize != 2 or not little_endian:
            raise TypeError(f"{pixel_format} requires standard little-endian uint16 containers")
    elif dtype != np.dtype(np.uint8):
        raise TypeError(f"{pixel_format} requires uint8 planes")


def _yuv_dtype(array: np.ndarray, pixel_format: str) -> None:
    _check_yuv_dtype(array.dtype, pixel_format)
    if pixel_format in {"P010", "P012"}:
        shift = 6 if pixel_format == "P010" else 4
        if np.any(np.bitwise_and(array, (1 << shift) - 1)):
            raise ValueError(f"{pixel_format} requires zero low {shift} padding bits")


def _yuv_dimensions(frame: ImageFrame, height: int, width: int, pixel_format: str) -> None:
    _check_dimensions(frame, height, width)
    if width % 2:
        raise ValueError(f"{pixel_format} requires an even width")
    if pixel_format not in {"YUYV", "UYVY"} and height % 2:
        raise ValueError(f"{pixel_format} requires an even height")


def _raw_yuv_planes(frame: ImageFrame, pixel_format: str) -> tuple[np.ndarray, ...]:
    width = _dimension(frame.width, "width", required=True)
    height = _dimension(frame.height, "height", required=True)
    _yuv_dimensions(frame, height, width, pixel_format)
    dtype = _declared_dtype(frame, required=True)
    _check_yuv_dtype(dtype, pixel_format)
    if pixel_format in {"YUYV", "UYVY"}:
        shapes = ((height, width * 2),)
    elif pixel_format in {"I420", "YV12"}:
        shapes = ((height, width), (height // 2, width // 2), (height // 2, width // 2))
    else:
        shapes = ((height, width), (height // 2, width))
    strides = _strides(frame, len(shapes), raw=True)
    if isinstance(frame.data, tuple):
        if len(frame.data) != len(shapes):
            raise ValueError(f"{pixel_format} requires {len(shapes)} raw plane buffers")
        if not all(isinstance(plane, (bytes, bytearray, memoryview)) for plane in frame.data):
            raise TypeError("Raw YUV plane tuples must contain only byte buffers")
        buffers = tuple(_buffer(plane) for plane in cast(tuple[ImageBuffer, ...], frame.data))
        separate = True
    elif isinstance(frame.data, (bytes, bytearray, memoryview)):
        buffers = (_buffer(frame.data),) * len(shapes)
        separate = False
    else:
        raise TypeError("Raw YUV data must be a byte buffer or a tuple of byte buffers")
    planes = []
    offset = 0
    for index, (rows, columns) in enumerate(shapes):
        plane = _raw_plane(buffers[index], 0 if separate else offset,
                           rows, columns, strides[index], dtype)
        planes.append(plane)
        offset += rows * strides[index]
    return tuple(planes)


def _array_yuv_planes(frame: ImageFrame, pixel_format: str) -> tuple[np.ndarray, ...]:
    data = frame.data
    if isinstance(data, tuple):
        if not data or not all(isinstance(plane, np.ndarray) for plane in data):
            raise TypeError("YUV plane tuples must contain only ndarrays or only raw byte buffers")
        data = cast(tuple[np.ndarray, ...], data)
        expected = 3 if pixel_format in {"I420", "YV12"} else 2
        if pixel_format in {"YUYV", "UYVY"}:
            raise ValueError(f"{pixel_format} uses one packed ndarray, not a plane tuple")
        if len(data) != expected:
            raise ValueError(f"{pixel_format} requires {expected} ndarray planes")
        if data[0].ndim != 2:
            raise ValueError("The Y plane must have HW dimensions")
        height, width = data[0].shape
        _yuv_dimensions(frame, height, width, pixel_format)
        planes = [data[0]]
        for plane in data[1:]:
            if pixel_format in {"I420", "YV12"}:
                if plane.shape != (height // 2, width // 2):
                    raise ValueError(f"{pixel_format} chroma planes must have shape (H/2, W/2)")
            elif plane.shape == (height // 2, width // 2, 2):
                # Channel-interleaved chroma may itself be a non-contiguous view.
                pass
            elif plane.shape != (height // 2, width):
                raise ValueError(f"{pixel_format} chroma plane must have shape (H/2, W) or (H/2, W/2, 2)")
            planes.append(plane)
        return tuple(planes)
    if not isinstance(data, np.ndarray):
        raise TypeError("YUV input must be an ndarray, a plane tuple or a declared raw byte buffer")
    if pixel_format in {"YUYV", "UYVY"}:
        if data.ndim == 3 and data.shape[2] == 2:
            height, width = data.shape[:2]
        elif data.ndim == 2 and data.shape[1] % 2 == 0:
            height, packed_width = data.shape
            width = packed_width // 2
        else:
            raise ValueError(f"{pixel_format} requires shape (H, W, 2) or (H, 2*W)")
        _yuv_dimensions(frame, height, width, pixel_format)
        return (data,)
    if data.ndim != 2 or data.shape[0] % 3:
        raise ValueError(f"Combined {pixel_format} arrays require shape (H*3/2, W)")
    height, width = data.shape[0] * 2 // 3, data.shape[1]
    _yuv_dimensions(frame, height, width, pixel_format)
    if pixel_format in {"I420", "YV12"}:
        if not data.flags.c_contiguous:
            raise ValueError(
                f"Combined {pixel_format} arrays must be contiguous without plane padding; "
                "use separate planes or raw bytes with explicit per-plane row_strides"
            )
        flat = data.reshape(-1)
        y_size, chroma_size = height * width, height * width // 4
        return (
            flat[:y_size].reshape(height, width),
            flat[y_size:y_size + chroma_size].reshape(height // 2, width // 2),
            flat[y_size + chroma_size:].reshape(height // 2, width // 2),
        )
    return (data[:height], data[height:])


def _yuv_rgb(frame: ImageFrame, pixel_format: str) -> np.ndarray:
    if _layout(frame) != "HWC":
        raise ValueError("YUV planes use their native HWC/HW layout; CHW is only supported for RGB images")
    if frame.value_range is not None:
        raise ValueError("YUV code ranges are defined by color_range; value_range is only for RGB/gray")
    if not isinstance(frame.color_matrix, str) or frame.color_matrix not in {"bt601", "bt709"}:
        raise ValueError("YUV frames require explicit color_matrix='bt601' or 'bt709'")
    if not isinstance(frame.color_range, str) or frame.color_range not in {"full", "limited"}:
        raise ValueError("YUV frames require explicit color_range='full' or 'limited'")
    raw = isinstance(frame.data, (bytes, bytearray, memoryview)) or (
        isinstance(frame.data, tuple) and frame.data
        and isinstance(frame.data[0], (bytes, bytearray, memoryview))
    )
    planes = _raw_yuv_planes(frame, pixel_format) if raw else _array_yuv_planes(frame, pixel_format)
    _check_plane_metadata(frame, planes)
    for plane in planes:
        _yuv_dtype(plane, pixel_format)
    bits = {"P010": 10, "P012": 12}.get(pixel_format, 8)
    if bits > 8:
        planes = tuple(np.right_shift(plane, 16 - bits) for plane in planes)
    if pixel_format in {"YUYV", "UYVY"}:
        packed = planes[0]
        if packed.ndim == 3:
            first, second = packed[..., 0], packed[..., 1]
            if pixel_format == "YUYV":
                y, u, v = first, second[:, 0::2], second[:, 1::2]
            else:
                y, u, v = second, first[:, 0::2], first[:, 1::2]
        elif pixel_format == "YUYV":
            y, u, v = packed[:, 0::2], packed[:, 1::4], packed[:, 3::4]
        else:
            y, u, v = packed[:, 1::2], packed[:, 0::4], packed[:, 2::4]
        u, v = u.repeat(2, axis=1), v.repeat(2, axis=1)
    else:
        y = planes[0]
        if pixel_format in {"I420", "YV12"}:
            u, v = planes[1:]
            if pixel_format == "YV12":
                u, v = v, u
        else:
            chroma = planes[1]
            if chroma.ndim == 3:
                u, v = chroma[..., 0], chroma[..., 1]
            else:
                u, v = chroma[:, 0::2], chroma[:, 1::2]
            if pixel_format == "NV21":
                u, v = v, u
        u = u.repeat(2, axis=0).repeat(2, axis=1)
        v = v.repeat(2, axis=0).repeat(2, axis=1)
    y, u, v = (plane.astype(np.float64) for plane in (y, u, v))
    if frame.color_range == "limited":
        scale = 1 << (bits - 8)
        y = (y - 16 * scale) / (219 * scale)
        u = (u - 128 * scale) / (224 * scale)
        v = (v - 128 * scale) / (224 * scale)
    else:
        maximum, center = (1 << bits) - 1, 1 << (bits - 1)
        y, u, v = y / maximum, (u - center) / maximum, (v - center) / maximum
    kr, kb = (0.299, 0.114) if frame.color_matrix == "bt601" else (0.2126, 0.0722)
    kg = 1.0 - kr - kb
    red = y + 2.0 * (1.0 - kr) * v
    blue = y + 2.0 * (1.0 - kb) * u
    green = y - 2.0 * kb * (1.0 - kb) / kg * u - 2.0 * kr * (1.0 - kr) / kg * v
    rgb = np.stack((red, green, blue), axis=2)
    return np.floor(np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _png_depth(header: bytes) -> int | None:
    if header[:8] != _PNG_SIGNATURE:
        return None
    if len(header) < 29 or header[8:12] != b"\x00\x00\x00\x0d" or header[12:16] != b"IHDR":
        raise ValueError("PNG input has a truncated or invalid IHDR header")
    return header[24]


def _decode_png16(encoded: bytes) -> np.ndarray:
    # Pillow converts RGB16/RGBA16 PNGs to 8 bits; use the existing OpenCV dependency.
    import cv2

    try:
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("Unable to decode 16-bit PNG input") from exc
    if decoded is None:
        raise ValueError("Unable to decode 16-bit PNG input")
    if decoded.dtype != np.dtype(np.uint16):
        raise ValueError("PNG declares 16-bit pixels but the decoder did not preserve uint16")
    if decoded.ndim == 2:
        return decoded
    if decoded.ndim != 3 or decoded.shape[2] not in {3, 4}:
        raise ValueError("Unsupported decoded 16-bit PNG channel layout")
    return decoded[..., [2, 1, 0, 3]] if decoded.shape[2] == 4 else decoded[..., ::-1]


def _pil_png_depth(image: Image.Image) -> int | None:
    """Inspect source metadata without changing the caller's PIL stream position."""
    for tile in getattr(image, "tile", ()):
        arguments = tile[3]
        raw_mode = arguments[0] if isinstance(arguments, tuple) else arguments
        if isinstance(raw_mode, str) and ";16" in raw_mode:
            return 16
    stream = getattr(image, "fp", None)
    if stream is not None and not stream.closed and stream.seekable():
        position = stream.tell()
        try:
            stream.seek(0)
            return _png_depth(stream.read(29))
        finally:
            stream.seek(position)
    filename = getattr(image, "filename", None)
    if filename:
        try:
            with Path(filename).open("rb") as source:
                return _png_depth(source.read(29))
        except OSError:
            # A detached PIL image may outlive its original source file.
            pass
    return None


def _pil_array(image: Image.Image, frame: ImageFrame) -> np.ndarray:
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Expected one still image; decode animation/video into individual frames first")
    if image.format == "PNG" and image.mode in {"RGB", "RGBA"}:
        depth = _pil_png_depth(image)
        if depth == 16:
            raise ValueError(
                "PIL RGB/RGBA PNG has already lost its 16-bit samples; pass the original path "
                "or encoded PNG bytes with value_range instead"
            )
        # Detached, already decoded PIL objects have only their current 8-bit
        # pixels; any unrecorded source bit depth cannot be reconstructed.
    if image.mode.startswith("I;16") or image.mode in {"I", "F"}:
        if frame.value_range is None:
            raise ValueError(f"PIL mode {image.mode!r} requires explicit value_range before quantization")
        return np.asarray(image)
    return np.asarray(image.convert("RGB"))


def _decoded_array(frame: ImageFrame) -> np.ndarray:
    if frame.pixel_format is not None:
        raise ValueError(
            "File/PIL/encoded-image pixel_format is determined by the decoder; "
            "omit pixel_format to avoid a second color-order conversion"
        )
    if _layout(frame) != "HWC":
        raise ValueError("File/PIL/encoded images have decoded HWC/HW layout; omit CHW")
    if frame.row_strides is not None:
        raise ValueError("row_strides describes raw memory or ndarray planes, not encoded/PIL images")
    data = frame.data
    if isinstance(data, Image.Image):
        return _pil_array(data, frame)
    if isinstance(data, (str, Path)):
        path = Path(data)
        with path.open("rb") as source:
            header = source.read(29)
            if _png_depth(header) == 16:
                source.seek(0)
                return _decode_png16(source.read())
        with Image.open(path) as image:
            return _pil_array(image, frame)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Encoded image data must be a path or byte buffer")
    encoded = bytes(_buffer(data))
    if _png_depth(encoded[:29]) == 16:
        return _decode_png16(encoded)
    with Image.open(io.BytesIO(encoded)) as image:
        return _pil_array(image, frame)


def to_rgb_uint8(image: ImageInput, *, copy: bool = True) -> np.ndarray:
    """Return contiguous RGB HWC uint8 without modifying the supplied image.

    The default returns an independent writable array. ``copy=False`` permits
    reuse of an already canonical contiguous RGB uint8 array, including a
    read-only array; callers using that path must treat the result as read-only.
    Non-default RGB code ranges require ImageFrame.value_range. YUV requires
    explicit color_matrix and color_range. Invalid frames raise rather than
    being substituted with a black image.

    Already decoded 8-bit PIL RGB/RGBA objects are interpreted as their current
    pixels. To preserve source PNG16 samples, pass the original path or encoded
    bytes: source bit depth already discarded by another decoder is unavailable.
    """
    if not isinstance(copy, bool):
        raise TypeError("copy must be a bool")
    frame = image if isinstance(image, ImageFrame) else ImageFrame(image)
    pixel_format = _pixel_format(frame.pixel_format)
    _layout(frame)
    _dimension(frame.width, "width")
    _dimension(frame.height, "height")
    _declared_dtype(frame)
    _value_range(frame)
    if pixel_format in _YUV_FORMATS:
        rgb = _yuv_rgb(frame, pixel_format)
    else:
        if frame.color_matrix is not None or frame.color_range is not None:
            raise ValueError("color_matrix/color_range apply only to explicitly declared YUV frames")
        if isinstance(frame.data, np.ndarray):
            array = frame.data
        elif isinstance(frame.data, (bytes, bytearray, memoryview)) and pixel_format is not None:
            array = _raw_rgb(frame, pixel_format)
        elif isinstance(frame.data, (str, Path, Image.Image, bytes, bytearray, memoryview)):
            array = _decoded_array(frame)
        else:
            raise TypeError("Image input must be an ndarray, PIL image, path, encoded bytes or declared YUV planes")
        rgb = _rgb_uint8(array, frame, pixel_format)
    if copy:
        return np.array(rgb, dtype=np.uint8, order="C", copy=True)
    return np.ascontiguousarray(rgb, dtype=np.uint8)
