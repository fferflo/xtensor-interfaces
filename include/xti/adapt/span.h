#pragma once

#include <xti/exception.h>
#include <type_traits>
#include <array>
#include <xtensor/xtensor.hpp>
#include <xti/typedefs.h>

namespace xti {

template <typename TElementType, typename TThisType>
class Span
{
public:
  struct allocator_type
  {
  };

  using reference = TElementType&;
  using const_reference = const TElementType&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = TElementType;
  using pointer = TElementType*;
  using const_pointer = const TElementType*;
  using iterator = TElementType*;
  using const_iterator = const TElementType*;
  using reverse_iterator = TElementType*;
  using const_reverse_iterator = const TElementType*;

  TElementType* data()
  {
    return static_cast<TThisType*>(this)->data();
  }

  const TElementType* data() const
  {
    return static_cast<const TThisType*>(this)->data();
  }

  size_t size() const
  {
    return static_cast<const TThisType*>(this)->size();
  }

  TElementType& operator[](int i)
  {
    return data()[i];
  }

  const TElementType& operator[](int i) const
  {
    return data()[i];
  }

  TElementType& front()
  {
    return data()[0];
  }

  TElementType& back()
  {
    return data()[size() - 1];
  }

  TElementType* begin()
  {
    return data();
  }

  TElementType* end()
  {
    return data() + size();
  }

  const TElementType* cbegin() const
  {
    return data();
  }

  const TElementType* cend() const
  {
    return data() + size();
  }

  void resize(size_t)
  {
    throw ResizeNotSupportedException();
  }
};

template <size_t TDims, xt::layout_type TLayout = xt::layout_type::row_major, typename TSpan, typename TShape>
auto construct_span_xtensor_container(TSpan&& span, TShape&& shape_in)
{
  std::array<unsigned long int, TDims> shape;
  std::copy(shape_in.begin(), shape_in.end(), shape.begin());
  std::array<long int, TDims> strides;
  xt::compute_strides(shape, TLayout, strides);
  return xt::xtensor_container<std::remove_reference_t<TSpan>, TDims, TLayout>(std::forward<TSpan>(span), std::move(shape), std::move(strides));
}

} // end of ns xti
