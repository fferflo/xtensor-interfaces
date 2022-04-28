#pragma once

#include <xti/util.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xiterator.hpp>
#include <xtensor/xstrides.hpp>
#include <array>
#include <type_traits>

namespace xti {
template <typename TElementType, typename TThisType>
class Functor;
} // end of ns xti

namespace xt {

template <typename TElementType, typename TThisType>
struct xcontainer_inner_types<xti::Functor<TElementType, TThisType>>
{
  using temporary_type = xt::xtensor<TElementType, 2>;
};

template <typename TElementType, typename TThisType>
struct xiterable_inner_types<xti::Functor<TElementType, TThisType>>
{
  using inner_shape_type = std::array<size_t, 2>;
  using stepper = xt::xindexed_stepper<xti::Functor<TElementType, TThisType>, false>;
  using const_stepper = xt::xindexed_stepper<xti::Functor<TElementType, TThisType>, true>;
};

} // end of ns xt

namespace xti {

template <typename TElementType, typename TThisType>
class Functor : public xt::xiterable<Functor<TElementType, TThisType>>,
                public xt::xcontainer_semantic<Functor<TElementType, TThisType>>
{
private:
  std::array<size_t, 2> m_shape;

public:
  using self_type = Functor<TElementType, TThisType>;
  using semantic_base = xt::xcontainer_semantic<self_type>;

  using value_type = TElementType;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using inner_shape_type = std::array<size_t, 2>;
  using inner_strides_type = inner_shape_type;
  using shape_type = inner_shape_type;
  using strides_type = inner_strides_type;

  using iterable_base = xt::xiterable<self_type>;
  using stepper = typename iterable_base::stepper;
  using const_stepper = typename iterable_base::const_stepper;

  using temporary_type = xt::xtensor<TElementType, 2>;

  using bool_load_type = xt::bool_load_type<value_type>;

  static constexpr xt::layout_type static_layout = xt::layout_type::dynamic;
  static constexpr bool contiguous_layout = false;

  template <typename TStrides>
  bool has_linear_assign(const TStrides&) const
  {
      return false;
  }

  bool is_contiguous() const
  {
    return false;
  }

  constexpr xt::layout_type layout() const
  {
      return static_layout;
  }

  Functor(const std::array<size_t, 2>& shape)
    : m_shape(shape)
  {
  }

  Functor(const Functor<TElementType, TThisType>& other)
    : m_shape(other.m_shape)
  {
  }

  Functor(Functor<TElementType, TThisType>&& other)
    : m_shape(other.m_shape)
  {
  }

  Functor<TElementType, TThisType>& operator=(const Functor<TElementType, TThisType>& other)
  {
    this->m_shape = other.m_shape;
    return *this;
  }

  Functor<TElementType, TThisType>& operator=(Functor<TElementType, TThisType>&& other)
  {
    this->m_shape = std::move(other.m_shape);
    return *this;
  }

  template <typename TExpression>
  self_type& operator=(const xt::xexpression<TExpression>& e)
  {
    return semantic_base::operator=(e);
  }

  self_type& operator=(const temporary_type& e)
  {
    xt::computed_assign(*this, e);
    return *this;
  }

  ~Functor()
  {
  }



  template <typename... TArgs>
  reference operator()(TArgs&&... args)
  {
    return static_cast<TThisType*>(this)->operator()(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  const_reference operator()(TArgs&&... args) const
  {
    return static_cast<const TThisType*>(this)->operator()(std::forward<TArgs>(args)...);
  }

  reference operator[](const xt::xindex& index)
  {
    return element(index.cbegin(), index.cend());
  }

  const_reference operator[](const xt::xindex& index) const
  {
    return element(index.cbegin(), index.cend());
  }

  reference operator[](size_type i)
  {
    return operator()(i);
  }

  const_reference operator[](size_type i) const
  {
    return operator()(i);
  }

  template <typename TIterator>
  reference element(TIterator first, TIterator last)
  {
    XTENSOR_TRY(xt::check_element_index(shape(), first, last));
    std::vector<size_t> coords(first, last);
    size_t num_coords = coords.size();
    if (num_coords == 1)
    {
      return operator()(0, coords[0]);
    }
    else
    {
      return operator()(coords[num_coords - 2], coords[num_coords - 1]);
    }
  }

  template <typename TIterator>
  const_reference element(TIterator first, TIterator last) const
  {
    XTENSOR_TRY(xt::check_element_index(shape(), first, last));
    std::vector<size_t> coords(first, last);
    size_t num_coords = coords.size();
    if (num_coords == 1)
    {
      return operator()(0, coords[0]);
    }
    else
    {
      return operator()(coords[num_coords - 2], coords[num_coords - 1]);
    }
  }



  size_type dimension() const
  {
    return 2;
  }

  const shape_type& shape() const
  {
    return m_shape;
  }

  size_t size() const
  {
    return m_shape[0] * m_shape[1];
  }

  template <typename TShape>
  bool broadcast_shape(TShape& s) const
  {
    return xt::broadcast_shape(shape(), s);
  }



  template <typename TShape = shape_type>
  auto resize(TShape&& shape, bool force = false)
  {
    throw ResizeNotSupportedException();
  }

  template <typename TShape = shape_type>
  auto resize(TShape&& shape, xt::layout_type l)
  {
    throw ResizeNotSupportedException();
  }

  template <typename TShape = shape_type>
  auto resize(TShape&& shape, const strides_type& strides)
  {
    throw ResizeNotSupportedException();
  }



  template <typename ST>
  stepper stepper_begin(const ST& s)
  {
    size_type offset = s.size() - dimension();
    return stepper(this, offset);
  }

  template <typename ST>
  stepper stepper_end(const ST& s)
  {
    size_type offset = s.size() - dimension();
    return stepper(this, offset, true);
  }

  template <typename ST>
  const_stepper stepper_begin(const ST& s) const
  {
    size_type offset = s.size() - dimension();
    return const_stepper(this, offset);
  }

  template <typename ST>
  const_stepper stepper_end(const ST& s) const
  {
    size_type offset = s.size() - dimension();
    return const_stepper(this, offset, true);
  }
};

} // end of ns xti
