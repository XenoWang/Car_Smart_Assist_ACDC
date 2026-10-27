"""Independent format evidence grounded in docs/camera_inputs.md.

Expected pixels are fixed literals or V4L2/ITU matrix calculations; none are
obtained by calling the converter under test.
"""

from __future__ import annotations

import struct
import io
import zlib

import numpy as np
import pytest
from PIL import Image

from car_smart_assist.data.image_input import ImageFrame, to_rgb_uint8


def test_rgb_channel_order_alpha_gray_and_default_copy_are_observable():
    rgb = np.array([[[0, 85, 170], [255, 170, 85]]], dtype=np.uint8)
    expected = np.array([[[0, 85, 170], [255, 170, 85]]], dtype=np.uint8)
    np.testing.assert_array_equal(to_rgb_uint8(rgb), expected)
    np.testing.assert_array_equal(to_rgb_uint8(ImageFrame(rgb[:, :, ::-1], pixel_format="BGR")), expected)
    rgba = np.concatenate((rgb, np.array([[[7], [9]]], dtype=np.uint8)), axis=2)
    np.testing.assert_array_equal(to_rgb_uint8(ImageFrame(rgba, pixel_format="RGBA")), expected)
    np.testing.assert_array_equal(to_rgb_uint8(ImageFrame(rgba[:, :, [2, 1, 0, 3]], pixel_format="BGRA")), expected)
    np.testing.assert_array_equal(
        to_rgb_uint8(ImageFrame(np.array([[0, 85, 255]], np.uint8), pixel_format="GRAY")),
        np.array([[[0, 0, 0], [85, 85, 85], [255, 255, 255]]], np.uint8),
    )
    copied = to_rgb_uint8(rgb)
    assert copied.dtype == np.uint8 and copied.flags.c_contiguous and not np.shares_memory(copied, rgb)
    copied[0, 0, 0] = 99
    assert rgb[0, 0, 0] == 0


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_frames_use_fixed_declared_range_and_half_up_quantization(dtype):
    frame = np.array([[[0.0, 85 / 255, 0.5], [2 / 3, 1.0, 0.0]]], dtype=dtype)
    expected = np.array([[[0, 85, 128], [170, 255, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(to_rgb_uint8(frame), expected)
    explicit = ImageFrame(frame * 255, value_range=(0, 255))
    np.testing.assert_array_equal(to_rgb_uint8(explicit), expected)


@pytest.mark.parametrize("dtype,upper", [(np.uint16, 1023), (np.uint16, 4095), (np.uint16, 65535)])
def test_declared_high_bit_depth_rgb_maps_fixed_codes_not_per_frame_extrema(dtype, upper):
    codes = np.array([0, upper // 3, upper * 2 // 3, upper], dtype=dtype).reshape(1, 4, 1)
    rgb = np.repeat(codes, 3, axis=2)
    expected_codes = np.array([0, 85, 170, 255], dtype=np.uint8).reshape(1, 4, 1)
    expected = np.repeat(expected_codes, 3, axis=2)
    np.testing.assert_array_equal(to_rgb_uint8(ImageFrame(rgb, value_range=(0, upper))), expected)
    dark = np.full((1, 2, 3), 2, dtype=np.uint16)
    np.testing.assert_array_equal(
        to_rgb_uint8(ImageFrame(dark, value_range=(0, 4095))), np.full((1, 2, 3), 0, np.uint8)
    )


def test_chw_big_endian_readonly_noncontiguous_and_negative_stride_preserve_pixels():
    chw = np.array([[[0, 255]], [[85, 170]], [[170, 85]]], dtype=np.uint8)
    expected = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    np.testing.assert_array_equal(to_rgb_uint8(ImageFrame(chw, layout="CHW")), expected)
    big_endian = np.array([[[0, 21845, 43690], [65535, 43690, 21845]]], dtype=">u2")
    np.testing.assert_array_equal(to_rgb_uint8(ImageFrame(big_endian, value_range=(0, 65535))), expected)
    source = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)[:, ::2, :]
    source.setflags(write=False)
    snapshot = source.copy()
    converted = to_rgb_uint8(source)
    assert converted.flags.c_contiguous and converted.flags.writeable
    np.testing.assert_array_equal(source, snapshot)
    np.testing.assert_array_equal(to_rgb_uint8(source[:, ::-1, :]), snapshot[:, ::-1, :])


@pytest.mark.parametrize(
    "pixels,frame,exception,reason",
    [
        (np.zeros((1, 1, 3), bool), ImageFrame, TypeError, "bool"),
        (np.zeros((1, 1, 3), np.complex64), ImageFrame, TypeError, "real"),
        (np.array([[[object(), object(), object()]]], dtype=object), ImageFrame, TypeError, "real"),
        (np.array([[[0.0, 0.0, np.nan]]]), ImageFrame, ValueError, "NaN|非有限"),
        (np.array([[[0.0, 0.0, np.inf]]]), ImageFrame, ValueError, "Inf|非有限"),
        (np.array([[[0, 0, 0, np.nan]]], dtype=np.float32), lambda x: ImageFrame(x, pixel_format="RGBA"), ValueError, "alpha|非有限"),
        (np.full((1, 1, 3), 1.01, np.float32), ImageFrame, ValueError, "outside|range"),
        (np.zeros((1, 1, 3), np.uint16), ImageFrame, ValueError, "value_range"),
        (np.zeros((1, 1, 3), np.uint8), lambda x: ImageFrame(x, pixel_format="BGR", width=2), ValueError, "width"),
        (np.zeros((1, 1, 2), np.uint8), lambda x: ImageFrame(x, pixel_format="RGB"), ValueError, "channels|通道"),
        (np.zeros((1, 1, 3), np.uint8), lambda x: ImageFrame(x, pixel_format="RAW10"), ValueError, "Unsupported|unsupported"),
        (np.zeros((1, 1, 3), np.uint8), lambda x: ImageFrame(x, pixel_format="RGB_HDR"), ValueError, "Unsupported|unsupported"),
        (np.zeros((1, 1, 3), np.uint8), lambda x: ImageFrame(x, layout="NCHW"), ValueError, "layout"),
    ],
)
def test_invalid_rgb_frame_is_rejected_for_its_declared_input_reason(pixels, frame, exception, reason):
    with pytest.raises(exception, match=reason):
        to_rgb_uint8(frame(pixels))


@pytest.mark.parametrize(
    "value_range,reason",
    [
        ((5, 5), "lower < upper"),
        ((0, float("inf")), "finite"),
        ((0, 256), "container limits"),
    ],
)
def test_invalid_code_range_metadata_is_rejected_for_its_specific_reason(value_range, reason):
    valid_rgb = np.array([[[0, 85, 170]]], np.uint8)
    with pytest.raises(ValueError, match=reason):
        to_rgb_uint8(ImageFrame(valid_rgb, value_range=value_range))


def _yuv_frame(fmt, planes, *, matrix="bt601", color_range="limited"):
    return ImageFrame(planes, pixel_format=fmt, color_matrix=matrix, color_range=color_range)


def _red_fixture(fmt):
    """2x2 red block; chroma is shared per 2x2 sample block."""
    y = np.array([[81, 81], [81, 81]], np.uint8)
    uv = np.array([[90, 240]], np.uint8)
    if fmt == "YUYV":
        packed = np.array([[[81, 90], [81, 240]], [[81, 90], [81, 240]]], np.uint8)
        return packed
    if fmt == "UYVY":
        return np.array([[[90, 81], [240, 81]], [[90, 81], [240, 81]]], np.uint8)
    if fmt == "NV12": return (y, uv)
    if fmt == "NV21": return (y, uv[:, ::-1].copy())
    if fmt == "I420": return (y, np.array([[90]], np.uint8), np.array([[240]], np.uint8))
    if fmt == "YV12": return (y, np.array([[240]], np.uint8), np.array([[90]], np.uint8))
    bits = 10 if fmt == "P010" else 12
    shift = 16 - bits
    factor = 1 << (bits - 8)
    yd = (y.astype(np.uint16) * factor) << shift
    uvd = (uv.astype(np.uint16) * factor) << shift
    return (yd.astype("<u2"), uvd.astype("<u2"))


@pytest.mark.parametrize("fmt", ["YUYV", "UYVY", "NV12", "NV21", "I420", "YV12", "P010", "P012"])
def test_supported_yuv_layouts_match_independent_bt601_limited_color_reference(fmt):
    # docs/camera_inputs.md names these modes. V4L2/ITU-R BT.601 limited
    # (81,90,240) hand-computes to RGB (254,0,0), with floor(x+0.5) rounding.
    actual = to_rgb_uint8(_yuv_frame(fmt, _red_fixture(fmt)))
    np.testing.assert_array_equal(actual, np.full((2, 2, 3), [254, 0, 0], np.uint8))


@pytest.mark.parametrize("fmt", ["NV12", "NV21", "I420", "YV12", "P010", "P012"])
def test_420_chroma_sampling_reconstructs_four_distinct_color_blocks(fmt):
    y = np.array(
        [[81, 81, 41, 41], [81, 81, 41, 41], [41, 41, 81, 81], [41, 41, 81, 81]],
        np.uint8,
    )
    u = np.array([[90, 240], [240, 90]], np.uint8)
    v = np.array([[240, 110], [110, 240]], np.uint8)
    if fmt in {"NV12", "NV21"}:
        uv = np.empty((2, 2, 2), np.uint8)
        if fmt == "NV12":
            uv[..., 0], uv[..., 1] = u, v
        else:
            uv[..., 0], uv[..., 1] = v, u
        planes = (y, uv)
    elif fmt in {"I420", "YV12"}:
        planes = (y, u, v) if fmt == "I420" else (y, v, u)
    else:
        bits = 10 if fmt == "P010" else 12
        factor, shift = 1 << (bits - 8), 16 - bits
        uv_codes = np.stack((u, v), axis=2)
        packed_uv = (uv_codes.astype(np.uint16) * factor << shift).astype("<u2")
        packed_y = (y.astype(np.uint16) * factor << shift).astype("<u2")
        planes = (packed_y, packed_uv)
    expected = np.array(
        [
            [[254, 0, 0], [254, 0, 0], [0, 0, 255], [0, 0, 255]],
            [[254, 0, 0], [254, 0, 0], [0, 0, 255], [0, 0, 255]],
            [[0, 0, 255], [0, 0, 255], [254, 0, 0], [254, 0, 0]],
            [[0, 0, 255], [0, 0, 255], [254, 0, 0], [254, 0, 0]],
        ],
        np.uint8,
    )
    actual = to_rgb_uint8(_yuv_frame(fmt, planes))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "matrix,color_range,yuv,expected",
    [
        ("bt601", "limited", (81, 90, 240), (254, 0, 0)),
        ("bt709", "limited", (63, 102, 240), (255, 1, 0)),
        ("bt601", "full", (76, 85, 255), (254, 0, 0)),
        ("bt709", "full", (54, 99, 255), (254, 0, 0)),
    ],
)
def test_yuv_matrix_and_quantization_range_select_distinct_fixed_reference(matrix, color_range, yuv, expected):
    yy, uu, vv = yuv
    packed = np.array([[[yy, uu], [yy, vv]], [[yy, uu], [yy, vv]]], np.uint8)
    frame = ImageFrame(packed, pixel_format="YUYV", color_matrix=matrix, color_range=color_range)
    np.testing.assert_array_equal(to_rgb_uint8(frame), np.full((2, 2, 3), expected, np.uint8))


@pytest.mark.parametrize("fmt,bits", [("P010", 10), ("P012", 12)])
def test_high_bit_full_range_preserves_low_codes_black_midpoint_and_white(fmt, bits):
    shift = 16 - bits
    maximum = (1 << bits) - 1
    midpoint = 1 << (bits - 1)
    # 3/1023 and 9/4095 both round to code 1 after scaling by 255.
    # Discarding meaningful low bits before conversion would incorrectly yield 0.
    low_code = 3 if bits == 10 else 9
    codes = np.array([[0, low_code], [maximum, midpoint]], np.uint16)
    y = (codes << shift).astype("<u2")
    uv = np.full((1, 2), midpoint << shift, dtype="<u2")
    frame = ImageFrame((y, uv), pixel_format=fmt, color_matrix="bt601", color_range="full")
    gray = np.array([[0, 1], [255, 128]], np.uint8)
    np.testing.assert_array_equal(to_rgb_uint8(frame), np.repeat(gray[..., None], 3, axis=2))


@pytest.mark.parametrize("fmt", ["YUYV", "UYVY", "NV12", "NV21", "I420", "YV12", "P010", "P012"])
def test_raw_yuv_rows_with_padding_do_not_shift_following_rows(fmt):
    clean = _red_fixture(fmt)
    dtype = np.dtype("<u2") if fmt in {"P010", "P012"} else np.dtype(np.uint8)
    planes = (clean,) if fmt in {"YUYV", "UYVY"} else clean
    shapes = [(2, 4)] if fmt in {"YUYV", "UYVY"} else [p.shape[:2] for p in planes]
    strides = tuple(shape[1] * dtype.itemsize + 3 * dtype.itemsize for shape in shapes)
    raw_planes = []
    for index, (plane, shape, stride) in enumerate(zip(planes, shapes, strides, strict=True)):
        rows, columns = shape
        sentinel = (17 + 61 * index) & (255 if dtype.itemsize == 1 else 65535)
        raw = bytearray()
        for row in range(rows):
            row_bytes = np.ascontiguousarray(plane[row]).tobytes()
            raw.extend(row_bytes)
            raw.extend(np.full(stride - len(row_bytes), sentinel, dtype=np.uint8).tobytes())
        raw_planes.append(bytes(raw))
    payload = b"".join(raw_planes)
    padded = ImageFrame(payload, pixel_format=fmt, width=2, height=2, dtype=dtype, row_strides=strides,
                        color_matrix="bt601", color_range="limited")
    np.testing.assert_array_equal(to_rgb_uint8(padded), np.full((2, 2, 3), [254, 0, 0], np.uint8))


@pytest.mark.parametrize("fmt", ["NV12", "NV21", "I420", "YV12", "P010", "P012"])
def test_yuv_array_and_separate_raw_plane_representations_are_equivalent(fmt):
    planes = _red_fixture(fmt)
    frame = _yuv_frame(fmt, planes)
    expected = np.full((2, 2, 3), [254, 0, 0], np.uint8)
    np.testing.assert_array_equal(to_rgb_uint8(frame), expected)
    dtype = np.dtype("<u2") if fmt in {"P010", "P012"} else np.dtype(np.uint8)
    buffers = tuple(np.ascontiguousarray(p).tobytes() for p in planes)
    strides = tuple(p.shape[1] * dtype.itemsize for p in planes)
    raw_frame = ImageFrame(buffers, pixel_format=fmt, width=2, height=2, dtype=dtype,
                           row_strides=strides, color_matrix="bt601", color_range="limited")
    np.testing.assert_array_equal(to_rgb_uint8(raw_frame), expected)


@pytest.mark.parametrize("fmt", ["NV12", "NV21", "I420", "YV12", "P010", "P012"])
def test_combined_420_array_layouts_match_literal_red_reference(fmt):
    planes = _red_fixture(fmt)
    combined = np.concatenate([plane.reshape(-1) for plane in planes]).reshape(3, 2)
    actual = to_rgb_uint8(_yuv_frame(fmt, combined))
    np.testing.assert_array_equal(actual, np.full((2, 2, 3), [254, 0, 0], np.uint8))


@pytest.mark.parametrize(
    "fmt,packed",
    [
        ("YUYV", np.array([[81, 90, 81, 240], [81, 90, 81, 240]], np.uint8)),
        ("YUY2", np.array([[81, 90, 81, 240], [81, 90, 81, 240]], np.uint8)),
        ("UYVY", np.array([[90, 81, 240, 81], [90, 81, 240, 81]], np.uint8)),
    ],
)
def test_combined_422_byte_rows_match_literal_red_reference(fmt, packed):
    actual = to_rgb_uint8(_yuv_frame(fmt, packed))
    np.testing.assert_array_equal(actual, np.full((2, 2, 3), [254, 0, 0], np.uint8))


def test_combined_i420_with_row_padding_is_rejected_but_strided_planes_are_accepted():
    combined_backing = np.array(
        [[81, 81, 201, 202], [81, 81, 203, 204], [90, 240, 205, 206]], np.uint8
    )
    combined = combined_backing[:, ::2]
    with pytest.raises(ValueError, match="contiguous without plane padding"):
        to_rgb_uint8(_yuv_frame("I420", combined))

    y_backing = np.full((2, 5), 33, np.uint8)
    u_backing = np.full((1, 5), 34, np.uint8)
    v_backing = np.full((1, 5), 35, np.uint8)
    y_backing[:, :2] = 81
    u_backing[0, 0] = 90
    v_backing[0, 0] = 240
    y, u, v = y_backing[:, :2], u_backing[:, :1], v_backing[:, :1]
    frame = ImageFrame(
        (y, u, v), pixel_format="I420", row_strides=(y.strides[0], u.strides[0], v.strides[0]),
        color_matrix="bt601", color_range="limited",
    )
    np.testing.assert_array_equal(to_rgb_uint8(frame), np.full((2, 2, 3), [254, 0, 0], np.uint8))


@pytest.mark.parametrize(
    "frame,expected",
    [
        (
            ImageFrame(
                bytes([170, 85, 0, 85, 170, 255, 238, 239]),
                pixel_format="BGR", width=2, height=1, dtype=np.uint8, row_strides=(8,),
            ),
            np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8),
        ),
        (
            ImageFrame(
                bytes([0, 255, 238, 239, 85, 170, 238, 239, 170, 85, 238, 239]),
                pixel_format="RGB", width=2, height=1, layout="CHW", dtype=np.uint8,
                row_strides=(4,),
            ),
            np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8),
        ),
        (
            ImageFrame(
                np.array([[[0, 21845, 43690,], [65535, 43690, 21845]]], dtype=">u2").tobytes() + b"\xaa\xbb",
                pixel_format="RGB", width=2, height=1, dtype=">u2", row_strides=(14,),
                value_range=(0, 65535),
            ),
            np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8),
        ),
        (
            ImageFrame(
                bytes([170, 85, 0, 85, 170, 255, 238, 239,
                       255, 0, 0, 0, 255, 0, 236, 237]),
                pixel_format="BGR", width=2, height=2, dtype=np.uint8, row_strides=(8,),
            ),
            np.array(
                [[[0, 85, 170], [255, 170, 85]],
                 [[0, 0, 255], [0, 255, 0]]], np.uint8,
            ),
        ),
    ],
)
def test_raw_rgb_layouts_ignore_padding_and_preserve_declared_pixels(frame, expected):
    original = frame.data
    actual = to_rgb_uint8(frame)
    np.testing.assert_array_equal(actual, expected)
    assert frame.data == original


@pytest.mark.parametrize(
    "frame,reason",
    [
        (ImageFrame(np.zeros((2, 2, 3), np.uint8), pixel_format="YUYV", color_matrix="bt601", color_range="limited"), "shape"),
        (ImageFrame((np.zeros((2, 2), np.uint8), np.zeros((1, 1), np.uint8)), pixel_format="NV12", color_matrix="bt601", color_range="limited"), "shape"),
        (ImageFrame((np.zeros((2, 2), np.uint8), np.zeros((1, 2), np.uint8)), pixel_format="NV12", color_range="limited"), "color_matrix"),
        (ImageFrame((np.zeros((2, 2), np.uint8), np.zeros((1, 2), np.uint8)), pixel_format="NV12", color_matrix="bt601"), "color_range"),
        (ImageFrame((np.zeros((2, 2), np.uint8), np.zeros((1, 2), np.uint8)), pixel_format="NV12", color_matrix="bt2020", color_range="full"), "color_matrix"),
        (ImageFrame(b"\0" * 16, pixel_format="NV12", width=2, height=2, row_strides=(2, 2), color_matrix="bt601", color_range="limited"), "dtype"),
        (ImageFrame((np.zeros((2, 2), np.uint8), np.zeros((1, 2), np.uint8)), pixel_format="NV12", row_strides=(2, 4), color_matrix="bt601", color_range="limited"), "row_strides"),
        (ImageFrame((np.zeros((2, 2), ">u2"), np.zeros((1, 2), ">u2")), pixel_format="P010", color_matrix="bt601", color_range="limited"), "little-endian"),
    ],
)
def test_malformed_yuv_is_rejected_for_the_specific_frame_contract(frame, reason):
    with pytest.raises((TypeError, ValueError), match=reason):
        to_rgb_uint8(frame)


@pytest.mark.parametrize(
    "frame,reason",
    [
        (ImageFrame(b"\0" * 5, pixel_format="NV12", width=2, height=2, dtype=np.uint8,
                    row_strides=(2, 2), color_matrix="bt601", color_range="limited"), "truncated"),
        (ImageFrame(b"\0" * 20, pixel_format="NV12", width=2, height=2, dtype=np.uint8,
                    row_strides=(1, 2), color_matrix="bt601", color_range="limited"), "stride"),
        (ImageFrame(np.zeros((2, 3, 2), np.uint8), pixel_format="YUYV",
                    color_matrix="bt601", color_range="limited"), "even width"),
        (ImageFrame((np.zeros((1, 2), np.uint8), np.zeros((1, 2), np.uint8)), pixel_format="NV12",
                    color_matrix="bt601", color_range="limited"), "even height"),
    ],
)
def test_raw_yuv_rejects_truncation_minimum_stride_and_odd_dimensions(frame, reason):
    with pytest.raises(ValueError, match=reason):
        to_rgb_uint8(frame)


@pytest.mark.parametrize("fmt,bits", [("P010", 10), ("P012", 12)])
def test_packed_high_bit_yuv_rejects_nonzero_padding_bits(fmt, bits):
    shift = 16 - bits
    y = np.full((2, 2), 16 << (bits - 8), dtype=np.uint16) << shift
    uv = np.full((1, 2), 128 << (bits - 8), dtype=np.uint16) << shift
    y[0, 0] |= 1
    frame = ImageFrame((y.astype("<u2"), uv.astype("<u2")), pixel_format=fmt,
                       color_matrix="bt601", color_range="limited")
    with pytest.raises(ValueError, match="low .*padding bits"):
        to_rgb_uint8(frame)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _gray16_png() -> bytes:
    # PNG grayscale16, filter 0, literal big-endian samples; stdlib-only fixture.
    raw = b"\x00" + struct.pack(">HHH", 0, 32768, 65535)
    ihdr = struct.pack(">IIBBBBB", 3, 1, 16, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw)) + _png_chunk(b"IEND", b"")


def _rgb16_png() -> bytes:
    raw = b"\x00" + struct.pack(">HHH", 0, 32768, 65535)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw)) + _png_chunk(b"IEND", b"")


def test_png16_gray_preserves_source_codes_until_declared_range_mapping():
    encoded = _gray16_png()
    actual = to_rgb_uint8(ImageFrame(encoded, value_range=(0, 65535)))
    expected = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], np.uint8)
    np.testing.assert_array_equal(actual, expected)


def test_loaded_pil_image_and_encoded_bytes_share_rgb_contract_without_closing_caller_handle():
    rgb = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    np.testing.assert_array_equal(to_rgb_uint8(image), rgb)
    assert image.getpixel((0, 0)) == (0, 85, 170)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_encoded_path_bytes_and_pil_views_keep_decoder_defined_rgb_order(tmp_path, image_format):
    source = np.full((8, 8, 3), [240, 20, 10], dtype=np.uint8)
    stream = io.BytesIO()
    options = {"quality": 100, "subsampling": 0} if image_format == "JPEG" else {}
    Image.fromarray(source).save(stream, format=image_format, **options)
    encoded = stream.getvalue()
    path = tmp_path / f"camera.{image_format.lower()}"
    path.write_bytes(encoded)
    with Image.open(io.BytesIO(encoded)) as decoded:
        from_bytes = to_rgb_uint8(encoded)
        from_pil = to_rgb_uint8(decoded)
    from_path = to_rgb_uint8(path)
    np.testing.assert_array_equal(from_bytes, from_pil)
    np.testing.assert_array_equal(from_bytes, from_path)
    assert from_bytes[..., 0].mean() > 200 and from_bytes[..., 2].mean() < 40


def test_pil_16_bit_png_is_rejected_with_explicit_depth_reason():
    image = Image.open(__import__("io").BytesIO(_rgb16_png()))
    try:
        with pytest.raises(ValueError) as refusal:
            to_rgb_uint8(image)
        reason = str(refusal.value).lower()
        assert "16-bit" in reason and "png" in reason
        assert "original path" in reason and "encoded png bytes" in reason
    finally:
        image.close()
