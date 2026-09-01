// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/bitmap.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "image/bitmap.h"

namespace neuralsuite {
namespace image {

void ToGrayscale(const Bitmap& in, std::vector<float>* out) {
  out->assign(static_cast<size_t>(in.width) * in.height, 0.0f);
  const int c = in.channels;
  for (size_t i = 0; i < out->size(); ++i) {
    const uint8_t* p = in.pixels.data() + i * c;
    float value;
    if (c >= 3) {
      value = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    } else {
      value = p[0];
    }
    if (c == 2 || c == 4) {
      const float alpha = p[c - 1] / 255.0f;
      value = value * alpha + 255.0f * (1.0f - alpha);
    }
    (*out)[i] = value / 255.0f;
  }
}

void Resize(const std::vector<float>& src, int src_w, int src_h, std::vector<float>* dst, int dst_w, int dst_h) {
  dst->assign(static_cast<size_t>(dst_w) * dst_h, 0.0f);
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

  // Se mapean centros de pixel, no esquinas: con esquinas la imagen se desplaza
  // medio pixel y el borde derecho se estira.
  const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
  const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    const float fy = std::max(0.0f, (y + 0.5f) * scale_y - 0.5f);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wy = fy - static_cast<float>(y0);

    for (int x = 0; x < dst_w; ++x) {
      const float fx = std::max(0.0f, (x + 0.5f) * scale_x - 0.5f);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, src_w - 1);
      const float wx = fx - static_cast<float>(x0);

      const float a = src[static_cast<size_t>(y0) * src_w + x0];
      const float b = src[static_cast<size_t>(y0) * src_w + x1];
      const float c = src[static_cast<size_t>(y1) * src_w + x0];
      const float d = src[static_cast<size_t>(y1) * src_w + x1];

      (*dst)[static_cast<size_t>(y) * dst_w + x] =
          a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy;
    }
  }
}

}  // namespace image
}  // namespace neuralsuite
