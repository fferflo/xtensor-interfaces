#pragma once

#include <xti/exception.h>
#include <xti/adapt/span.h>
#include <xti/adapt/closure.h>
#include <opencv2/opencv.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xti/util.h>

namespace xti {

namespace opencv {

template <typename TPixelType>
struct pixeltype
{
  static_assert(std::is_same<TPixelType, void>::value, "Invalid pixel type");
};

#define XTI_OPENCV_PIXEL_TYPE(TYPE, VALUE_PREFIX) \
  template <> \
  struct pixeltype<TYPE> \
  { \
    static const int get(int channels) \
    { \
      switch (channels) \
      { \
        case 1: return VALUE_PREFIX ## 1; \
        case 2: return VALUE_PREFIX ## 2; \
        case 3: return VALUE_PREFIX ## 3; \
        case 4: return VALUE_PREFIX ## 4; \
        default: throw InvalidDimensionException("Can only convert tensors with 1 to 4 channels to opencv, got " + std::to_string(channels) + " channels"); \
      } \
    } \
  };

XTI_OPENCV_PIXEL_TYPE(uint8_t, CV_8UC)
XTI_OPENCV_PIXEL_TYPE(int8_t, CV_8SC)
XTI_OPENCV_PIXEL_TYPE(uint16_t, CV_16UC)
XTI_OPENCV_PIXEL_TYPE(int16_t, CV_16SC)
XTI_OPENCV_PIXEL_TYPE(int32_t, CV_32SC)
XTI_OPENCV_PIXEL_TYPE(float, CV_32FC)
XTI_OPENCV_PIXEL_TYPE(double, CV_64FC)
// Not supported:
// XTI_OPENCV_PIXEL_TYPE(uint32_t, CV_32UC)
// XTI_OPENCV_PIXEL_TYPE(uint64_t, CV_64UC)
// XTI_OPENCV_PIXEL_TYPE(int64_t, CV_64SC)

#undef XTI_OPENCV_PIXEL_TYPE

template <typename TCvMat, typename TElementType>
class Span : public xti::Span<TElementType, Span<TCvMat, TElementType>>
           , public Closure<TCvMat>
{
public:
  Span(TCvMat&& mat)
    : Closure<TCvMat>(static_cast<TCvMat&&>(mat))
  {
    size_t elementsize = this->m_object.elemSize() / this->m_object.channels();
    if (elementsize != sizeof(TElementType))
    {
      throw InvalidElementTypeException(elementsize, sizeof(TElementType));
    }
  }

  TElementType* data()
  {
    return reinterpret_cast<TElementType*>(this->m_object.data);
  }

  const TElementType* data() const
  {
    return reinterpret_cast<const TElementType*>(this->m_object.data);
  }

  size_t size() const
  {
    return this->m_object.rows * this->m_object.cols * this->m_object.channels();
  }
};

} // end of ns opencv

template <typename TElementType, typename TCvMat>
auto from_opencv(TCvMat&& mat)
{
  xti::vec3i shape({mat.rows, mat.cols, mat.channels()});
  return construct_span_xtensor_container<3, xt::layout_type::row_major>(
    opencv::Span<TCvMat, TElementType>(std::forward<TCvMat>(mat)), shape
  );
}

template <typename TTensor>
cv::Mat to_opencv(TTensor&& tensor)
{
  using ElementType = typename std::decay<decltype(tensor())>::type;

  if constexpr (has_data_ptr_v<TTensor&&>)
  {
    if constexpr (layout_v<TTensor> == xt::layout_type::row_major)
    {
      cv::Mat result;
      if (tensor.dimension() == 2)
      {
        result = cv::Mat(tensor.shape()[0], tensor.shape()[1], opencv::pixeltype<ElementType>::get(1), const_cast<ElementType*>(tensor.data()));
      }
      else if (tensor.dimension() == 3)
      {
        result = cv::Mat(tensor.shape()[0], tensor.shape()[1], opencv::pixeltype<ElementType>::get(tensor.shape()[2]), const_cast<ElementType*>(tensor.data()));
      }
      else
      {
        throw InvalidDimensionException("Can only convert tensors with 2 or 3 dimensions to opencv, got shape " + XTI_TO_STRING(xt::adapt(tensor.shape())));
      }

      using Pointer = decltype(tensor.data());
      if constexpr (std::is_const_v<std::remove_pointer_t<Pointer>> || std::is_rvalue_reference_v<TTensor&&>)
      {
        result = result.clone();
      }
      return result;
    }
  }

  cv::Mat result(tensor.shape()[0], tensor.shape()[1], opencv::pixeltype<ElementType>::get(tensor.shape()[2]));
  from_opencv<ElementType>(result).assign(std::forward<TTensor>(tensor));
  return result;
}

} // end of ns xti
