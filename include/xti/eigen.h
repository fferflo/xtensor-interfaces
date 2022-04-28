#pragma once

#include <xti/exception.h>
#include <xti/util.h>
#include <xti/adapt/functor.h>
#include <xti/adapt/closure.h>
#include <xti/adapt/span.h>
#include <Eigen/Dense>
#include <xtensor/xtensor.hpp>
#include <xtensor/xiterator.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor/xadapt.hpp>
#include <utility>
#include <type_traits>

namespace xti::eigen {

template <typename TTensor>
struct has_data_ptr
{
  template <typename TTensor2, typename = decltype(std::declval<TTensor2&>().data())>
  static std::true_type deduce(TTensor2&&);

  static std::false_type deduce(...);

  static const bool value = decltype(deduce(std::declval<TTensor>()))::value;
};

template <typename TTensor>
constexpr bool has_data_ptr_v = has_data_ptr<TTensor>::value;

template <typename TEigenMatrix>
using elementtype_t = std::remove_reference_t<decltype(std::declval<TEigenMatrix>()(0, 0))>;

template <typename TEigenMatrix>
class Functor;

} // end of ns xti::eigen

namespace xt {

template <typename TEigenMatrix>
struct xcontainer_inner_types<xti::eigen::Functor<TEigenMatrix>>
{
  using temporary_type = xt::xtensor<xti::eigen::elementtype_t<TEigenMatrix>, 2>;
};

template <typename TEigenMatrix>
struct xiterable_inner_types<xti::eigen::Functor<TEigenMatrix>>
{
  using inner_shape_type = std::array<size_t, 2>;
  using stepper = xt::xindexed_stepper<xti::eigen::Functor<TEigenMatrix>, false>;
  using const_stepper = xt::xindexed_stepper<xti::eigen::Functor<TEigenMatrix>, true>;
};

} // end of ns xt

namespace xti {
namespace eigen {

template <typename TEigenMatrix>
class Functor : public xti::Functor<eigen::elementtype_t<TEigenMatrix>, Functor<TEigenMatrix>>
              , public Closure<TEigenMatrix>
{
public:
  using FunctorBase = xti::Functor<eigen::elementtype_t<TEigenMatrix>, Functor<TEigenMatrix>>;
  using ElementType = eigen::elementtype_t<TEigenMatrix>;

  Functor(TEigenMatrix&& matrix)
    : FunctorBase({static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols())})
    , Closure<TEigenMatrix>(static_cast<TEigenMatrix&&>(matrix))
  {
  }

  using FunctorBase::operator=;

  template <typename... TArgs>
  ElementType& operator()(TArgs... args)
  {
    return this->m_object(get_coordinate<0>(args...), get_coordinate<1>(args...));
  }

  template <typename... TArgs>
  const ElementType& operator()(TArgs... args) const
  {
    return this->m_object(get_coordinate<0>(args...), get_coordinate<1>(args...));
  }
};

template <typename TEigenMatrix>
class Span : public xti::Span<eigen::elementtype_t<TEigenMatrix>, Span<TEigenMatrix>>
           , public Closure<TEigenMatrix>
{
public:
  using ElementType = eigen::elementtype_t<TEigenMatrix>;

  Span(TEigenMatrix&& mat)
    : Closure<TEigenMatrix>(static_cast<TEigenMatrix&&>(mat))
  {
  }

  ElementType* data()
  {
    return reinterpret_cast<ElementType*>(this->m_object.data());
  }

  const ElementType* data() const
  {
    return reinterpret_cast<const ElementType*>(this->m_object.data());
  }

  size_t size() const
  {
    return this->m_object.rows() * this->m_object.cols();
  }
};

} // end of ns eigen

constexpr xt::layout_type from_eigen_layout(int eigen_layout)
{
  if ((eigen_layout & Eigen::RowMajor == 0) && (eigen_layout & Eigen::ColMajor == 0))
  {
    throw InvalidLayoutException();
  }
  return eigen_layout & Eigen::RowMajor != 0 ? xt::layout_type::row_major : xt::layout_type::column_major;
}

template <typename TEigenMatrix>
constexpr xt::layout_type from_eigen_layout()
{
  return from_eigen_layout(std::decay_t<TEigenMatrix>::Flags);
}

constexpr int to_eigen_layout(xt::layout_type xtensor_layout)
{
  if (xtensor_layout != xt::layout_type::row_major && xtensor_layout != xt::layout_type::column_major)
  {
    throw InvalidLayoutException();
  }
  return xtensor_layout == xt::layout_type::row_major ? Eigen::RowMajor : Eigen::ColMajor;
}

template <typename TEigenMatrix>
auto from_eigen(TEigenMatrix&& matrix)
{
  if constexpr (eigen::has_data_ptr_v<TEigenMatrix&&>)
  {
    xti::vec2T<long int> shape({matrix.rows(), matrix.cols()});
    return construct_span_xtensor_container<2, from_eigen_layout<TEigenMatrix>()>(
      eigen::Span<TEigenMatrix>(std::forward<TEigenMatrix>(matrix)), shape
    );
  }
  else
  {
    return eigen::Functor<TEigenMatrix>(std::forward<TEigenMatrix>(matrix));
  }
}

template <typename TTensor>
auto to_eigen(TTensor&& xtensor)
{
  if (xtensor.dimension() != 2)
  {
    throw InvalidDimensionException("Can only convert tensors with 2 dimensions to eigen, got shape " + XTI_TO_STRING(xt::adapt(xtensor.shape())));
  }
  if constexpr (has_data_ptr_v<TTensor&&>)
  {
    using EigenMatrix = Eigen::Matrix<xti::elementtype_t<TTensor&&>, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(layout_v<TTensor>)>;
    auto result = Eigen::Map<EigenMatrix>(xtensor.data(), xtensor.shape()[0], xtensor.shape()[1]);
    if constexpr (std::is_rvalue_reference_v<TTensor&&>)
    {
      return EigenMatrix(result);
    }
    else
    {
      return result;
    }
  }
  else
  {
    using EigenMatrix = Eigen::Matrix<xti::elementtype_t<TTensor&&>, Eigen::Dynamic, Eigen::Dynamic>;
    EigenMatrix result(xtensor.shape()[0], xtensor.shape()[1]);
    from_eigen(result).assign(std::forward<TTensor>(xtensor));

    return result;
  }
}

} // end of ns xti
