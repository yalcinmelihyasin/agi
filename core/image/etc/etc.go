// Copyright (C) 2021 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package etc implements etc texture compression and decompression.
//
// etc is in a separate package from image as it contains cgo code that can
// slow builds.
package etc

/*
#include "etc2.h"
#include "stdlib.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/google/gapid/core/image"
)

var (
	// ETC2
	ETC2_RGB_U8_NORM         = image.NewETC2_RGB_U8_NORM("ETC2_RGB_U8_NORM")
	ETC2_RGBA_U8_NORM        = image.NewETC2_RGBA_U8_NORM("ETC2_RGBA_U8_NORM")
	ETC2_RGBA_U8U8U8U1_NORM  = image.NewETC2_RGBA_U8U8U8U1_NORM("ETC2_RGBA_U8U8U8U1_NORM")
	ETC2_SRGB_U8_NORM        = image.NewETC2_SRGB_U8_NORM("ETC2_SRGB_U8_NORM")
	ETC2_SRGBA_U8_NORM       = image.NewETC2_SRGBA_U8_NORM("ETC2_SRGBA_U8_NORM")
	ETC2_SRGBA_U8U8U8U1_NORM = image.NewETC2_SRGBA_U8U8U8U1_NORM("ETC2_SRGBA_U8U8U8U1_NORM")

	// EAC
	ETC2_R_U11_NORM  = image.NewETC2_R_U11_NORM("ETC2_R_U11_NORM")
	ETC2_RG_U11_NORM = image.NewETC2_RG_U11_NORM("ETC2_RG_U11_NORM")
	ETC2_R_S11_NORM  = image.NewETC2_R_S11_NORM("ETC2_R_S11_NORM")
	ETC2_RG_S11_NORM = image.NewETC2_RG_S11_NORM("ETC2_RG_S11_NORM")

	// ETC 1
	ETC1_RGB_U8_NORM = image.NewETC1_RGB_U8_NORM("ETC1_RGB_U8_NORM")
	formatToCEnum = map[interface{}]C.enum_etc_format{
		ETC2_RGB_U8_NORM: C.ETC2_RGB_U8_NORM,
		ETC2_RGBA_U8_NORM: C.ETC2_RGBA_U8_NORM,
		ETC2_RGBA_U8U8U8U1_NORM: C.ETC2_RGBA_U8U8U8U1_NORM,
		ETC2_SRGB_U8_NORM: C.ETC2_SRGB_U8_NORM,
		ETC2_SRGBA_U8_NORM: C.ETC2_SRGBA_U8_NORM,
		ETC2_SRGBA_U8U8U8U1_NORM: C.ETC2_SRGBA_U8U8U8U1_NORM,
		ETC2_R_U11_NORM: C.ETC2_R_U11_NORM,
		ETC2_RG_U11_NORM: C.ETC2_RG_U11_NORM,
		ETC2_R_S11_NORM: C.ETC2_R_S11_NORM,
		ETC2_RG_S11_NORM: C.ETC2_RG_S11_NORM,
		ETC1_RGB_U8_NORM: C.ETC1_RGB_U8_NORM,
	}
)

type converterLayout struct {
	uncompressed      *image.Format
	compressed *image.Format
}

type etcLayout struct {
	converterLayout
	alphaMode etcAlphaMode
}

type eacLayout struct {
	converterLayout
	channels int
}

func init() {
	//ETC2 Formats
	etc2SupportMap := []etcLayout{
		{converterLayout{image.RGBA_U8_NORM, ETC2_RGB_U8_NORM}, etcAlphaNone},
		{converterLayout{image.RGBA_U8_NORM, ETC2_RGBA_U8_NORM}, etcAlpha8Bit},
		{converterLayout{image.RGBA_U8_NORM, ETC2_RGBA_U8U8U8U1_NORM}, etcAlpha1Bit},
		{converterLayout{image.SRGBA_U8_NORM, ETC2_SRGB_U8_NORM}, etcAlphaNone},
		{converterLayout{image.SRGBA_U8_NORM, ETC2_SRGBA_U8_NORM}, etcAlpha8Bit},
		{converterLayout{image.SRGBA_U8_NORM, ETC2_SRGBA_U8U8U8U1_NORM}, etcAlpha1Bit},
	}

	for _, conversion := range etc2SupportMap {
		// Intentional local copy
		conv := conversion
		image.RegisterConverter(conv.compressed, conv.uncompressed, func(src []byte, w, h, d int) ([]byte, error) {
			return decodeETC(src, w, h, d, conv.alphaMode)
		})

		image.RegisterConverter(conv.uncompressed, conv.compressed, func(src []byte, w, h, d int) ([]byte, error) {
			bytesPerPixel := 4
			return compress(src, w, h, d, conv.compressed, bytesPerPixel)
		})
	}

	// EAC formats
	etcU11SupportMap := []eacLayout{
		{converterLayout{image.R_U16_NORM, ETC2_R_U11_NORM}, 1},
		{converterLayout{image.RG_U16_NORM, ETC2_RG_U11_NORM}, 2},
	}

	for _, conversion := range etcU11SupportMap {
		// Intentional local copy
		conv := conversion
		image.RegisterConverter(conv.compressed, conv.uncompressed, func(src []byte, w, h, d int) ([]byte, error) {
			return decodeETCU11(src, w, h, d, conv.channels)
		})

		image.RegisterConverter(conv.uncompressed, conv.compressed, func(src []byte, w, h, d int) ([]byte, error) {
			bytesPerPixel := conv.channels * 2
			return compress(src, w, h, d, conv.compressed, bytesPerPixel)
		})
	}

	etcS11SupportMap := []eacLayout{
		{converterLayout{image.R_S16_NORM, ETC2_R_S11_NORM}, 1},
		{converterLayout{image.RG_S16_NORM, ETC2_RG_S11_NORM}, 2},
	}

	for _, conversion := range etcS11SupportMap {
		// Intentional local copy
		conv := conversion
		image.RegisterConverter(conv.compressed, conv.uncompressed, func(src []byte, w, h, d int) ([]byte, error) {
			return decodeETCS11(src, w, h, d, conv.channels)
		})

		image.RegisterConverter(conv.uncompressed, conv.compressed, func(src []byte, w, h, d int) ([]byte, error) {
			bytesPerPixel := conv.channels * 2
			return compress(src, w, h, d, conv.compressed, bytesPerPixel)
		})
	}

	// ETC1 formats
	etc1SupportMap := []converterLayout{
		{image.RGB_U8_NORM, ETC1_RGB_U8_NORM},
		{image.RGBA_U8_NORM, ETC1_RGB_U8_NORM},
	}

	for _, conversion := range etc1SupportMap {
		conv := conversion
		image.RegisterConverter(conv.compressed, conv.uncompressed, func(src []byte, w, h, d int) ([]byte, error) {
			return image.Convert(src, w, h, d, ETC2_RGB_U8_NORM, conv.uncompressed)
		})

		image.RegisterConverter(conv.uncompressed, conv.compressed, func(src []byte, w, h, d int) ([]byte, error) {
			bytesPerPixel := 4
			return compress(src, w, h, d, conv.compressed, bytesPerPixel)
		})
	}

	// This is for converting via intermediate format.
	// TODO (melihyalcin): We should check if we really need all the conversion formats
	// especially from SRGB(A) formats to RGB(A) formats
	streamETCSupportMap := []converterLayout{
		{image.RGBA_U8_NORM, ETC2_RGB_U8_NORM},
		{image.RGBA_U8_NORM, ETC2_RGBA_U8_NORM},
		{image.RGBA_U8_NORM, ETC2_RGBA_U8U8U8U1_NORM},
		{image.SRGBA_U8_NORM, ETC2_SRGB_U8_NORM},
		{image.SRGBA_U8_NORM, ETC2_SRGBA_U8_NORM},
		{image.SRGBA_U8_NORM, ETC2_SRGBA_U8U8U8U1_NORM},
		{image.R_U16_NORM, ETC2_R_U11_NORM},
		{image.RG_U16_NORM, ETC2_RG_U11_NORM},
		{image.R_S16_NORM, ETC2_R_S11_NORM},
		{image.RG_S16_NORM, ETC2_RG_S11_NORM},
	}

	for _, conversion := range streamETCSupportMap {
		// Intentional local copy
		conv := conversion
		if !image.Registered(conv.compressed, image.RGB_U8_NORM) {
			image.RegisterConverter(conv.compressed, image.RGB_U8_NORM, func(src []byte, w, h, d int) ([]byte, error) {
				rgb, err := image.Convert(src, w, h, d, conv.compressed, conv.uncompressed)
				if err != nil {
					return nil, err
				}
				return image.Convert(rgb, w, h, d, conv.uncompressed, image.RGB_U8_NORM)
			})
		}
		if !image.Registered(conv.compressed, image.RGBA_U8_NORM) {
			image.RegisterConverter(conv.compressed, image.RGBA_U8_NORM, func(src []byte, w, h, d int) ([]byte, error) {
				rgba, err := image.Convert(src, w, h, d, conv.compressed, conv.uncompressed)
				if err != nil {
					return nil, err
				}
				return image.Convert(rgba, w, h, d, conv.uncompressed, image.RGBA_U8_NORM)
			})
		}

		if !image.Registered(image.RGB_U8_NORM, conv.compressed) {
			image.RegisterConverter(image.RGB_U8_NORM, conv.compressed, func(src []byte, w, h, d int) ([]byte, error) {
				rgb, err := image.Convert(src, w, h, d, image.RGB_U8_NORM, conv.uncompressed)
				if err != nil {
					return nil, err
				}
				return image.Convert(rgb, w, h, d, conv.uncompressed, conv.compressed)
			})
		}

		if !image.Registered(image.RGBA_U8_NORM, conv.compressed) {
			image.RegisterConverter(image.RGBA_U8_NORM, conv.compressed, func(src []byte, w, h, d int) ([]byte, error) {
				rgb, err := image.Convert(src, w, h, d, image.RGBA_U8_NORM, conv.uncompressed)
				if err != nil {
					return nil, err
				}
				return image.Convert(rgb, w, h, d, conv.uncompressed, conv.compressed)
			})
		}
	}
}

func compress(src []byte, width, height, depth int, format *image.Format, bytesPerPixel int) ([]byte, error) {
	dstSliceSize := format.Size(width, height, 1)
	srcSliceSize := width * height * bytesPerPixel
	dst := make([]byte, dstSliceSize*depth)

	for z := 0; z < depth; z++ {
		currentSrcSlice := src[srcSliceSize*z:]
		currentDstSlice := dst[dstSliceSize*z:]
		inputImageData := (unsafe.Pointer)(&currentSrcSlice[0])
		outputImageData := (unsafe.Pointer)(&currentDstSlice[0])

		result := C.compress_etc(
			(*C.uint8_t)(inputImageData),
			(*C.uint8_t)(outputImageData),
			(C.uint32_t)(width),
			(C.uint32_t)(height),
			(C.enum_etc_format)(formatToCEnum[format]),
		)

		if result != 0 {
			errorCString := C.get_etc_error_string(result)
			errorString := C.GoString(errorCString)
			C.free((unsafe.Pointer)(errorCString))
			return nil, fmt.Errorf("ETC Compression Status: %s", errorString)
		}
	}

	return dst, nil;
}
