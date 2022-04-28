#pragma once

#include <xtensor/xfixed.hpp>

namespace xti {

template <typename TElementType, size_t TDims>
using vecXT = xt::xtensor_fixed<TElementType, xt::xshape<TDims>>;

#define XTI_VECXT_T(DIMS) \
  template <typename TElementType> \
  using vec ## DIMS ## T = vecXT<TElementType, DIMS>

#define XTI_VECXT_X(TYPE_SHORT, TYPE_LONG) \
  template <size_t TDims> \
  using vecX ## TYPE_SHORT = vecXT<TYPE_LONG, TDims>

#define XTI_VECXT_XT(DIMS, TYPE_SHORT, TYPE_LONG) \
  using vec ## DIMS ## TYPE_SHORT = vecXT<TYPE_LONG, DIMS>

#define XTI_VECXT(TYPE_SHORT, TYPE_LONG) \
  XTI_VECXT_X(TYPE_SHORT, TYPE_LONG); \
  XTI_VECXT_T(1); \
  XTI_VECXT_T(2); \
  XTI_VECXT_T(3); \
  XTI_VECXT_T(4); \
  XTI_VECXT_XT(1, TYPE_SHORT, TYPE_LONG); \
  XTI_VECXT_XT(2, TYPE_SHORT, TYPE_LONG); \
  XTI_VECXT_XT(3, TYPE_SHORT, TYPE_LONG); \
  XTI_VECXT_XT(4, TYPE_SHORT, TYPE_LONG)

XTI_VECXT(i, int);
XTI_VECXT(u, unsigned int);
XTI_VECXT(s, size_t);
XTI_VECXT(f, float);
XTI_VECXT(d, double);


template <typename TElementType, size_t TRowsCols>
using matXT = xt::xtensor_fixed<TElementType, xt::xshape<TRowsCols, TRowsCols>>;

#define XTI_MATXT_T(ROWSCOLS) \
  template <typename TElementType> \
  using mat ## ROWSCOLS ## T = matXT<TElementType, ROWSCOLS>

#define XTI_MATXT_X(TYPE_SHORT, TYPE_LONG) \
  template <size_t TRowsCols> \
  using matX ## TYPE_SHORT = matXT<TYPE_LONG, TRowsCols>

#define XTI_MATXT_XT(ROWSCOLS, TYPE_SHORT, TYPE_LONG) \
  using mat ## ROWSCOLS ## TYPE_SHORT = matXT<TYPE_LONG, ROWSCOLS>

#define XTI_MATXT(TYPE_SHORT, TYPE_LONG) \
  XTI_MATXT_X(TYPE_SHORT, TYPE_LONG); \
  XTI_MATXT_T(1); \
  XTI_MATXT_T(2); \
  XTI_MATXT_T(3); \
  XTI_MATXT_T(4); \
  XTI_MATXT_XT(1, TYPE_SHORT, TYPE_LONG); \
  XTI_MATXT_XT(2, TYPE_SHORT, TYPE_LONG); \
  XTI_MATXT_XT(3, TYPE_SHORT, TYPE_LONG); \
  XTI_MATXT_XT(4, TYPE_SHORT, TYPE_LONG)

XTI_MATXT(i, int);
XTI_MATXT(u, unsigned int);
XTI_MATXT(s, size_t);
XTI_MATXT(f, float);
XTI_MATXT(d, double);

} // end of ns xti
