#pragma once

#include <utility>
#include <type_traits>
#include <sstream>
#include <xtensor/xio.hpp>

namespace xti {

template <typename TTensor>
struct is_xtensor
{
  template <typename TType>
  static std::true_type deduce(const xt::xexpression<TType>&);

  static std::false_type deduce(...);

  static const bool value = decltype(deduce(std::declval<TTensor>()))::value;
};

template <typename TTensor>
constexpr bool is_xtensor_v = is_xtensor<TTensor>::value;

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

template <typename TTensor>
constexpr xt::layout_type layout_v = xt::xcontainer_inner_types<std::decay_t<TTensor>>::layout;

#define XTI_TO_STRING(...) (static_cast<std::stringstream&&>((std::stringstream() << __VA_ARGS__)).str())

template <typename TTensor>
using elementtype_t = std::remove_reference_t<decltype(std::declval<TTensor>()())>;

template <size_t I, typename TArg0, typename... TArgs>
auto get_coordinate(TArg0 arg0, TArgs... args)
{
  if constexpr (I == 0)
  {
    return arg0;
  }
  else
  {
    return get_coordinate<I - 1>(args...);
  }
}

template <size_t I>
auto get_coordinate()
{
  return 0;
}

} // end of ns xti
