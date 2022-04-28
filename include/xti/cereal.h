#pragma once

#include <cereal/access.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>

namespace cereal {

template <typename TArchive, typename TElementType, size_t... TDims>
void save(TArchive& archive, const xt::xtensor_fixed<TElementType, xt::xshape<TDims...>>& tensor)
{
  for (const TElementType& el : xt::flatten(tensor))
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType, size_t... TDims>
void load(TArchive& archive, xt::xtensor_fixed<TElementType, xt::xshape<TDims...>>& tensor)
{
  for (TElementType& el : xt::flatten(tensor))
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType, size_t TRank>
void save(TArchive& archive, const xt::xtensor<TElementType, TRank>& tensor)
{
  for (size_t el : tensor.shape())
  {
    archive(el);
  }
  for (const TElementType& el : xt::flatten(tensor))
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType, size_t TRank>
void load(TArchive& archive, xt::xtensor<TElementType, TRank>& tensor)
{
  std::array<size_t, TRank> shape;
  for (size_t& el : shape)
  {
    archive(el);
  }
  tensor = xt::xtensor<TElementType, TRank>(shape);
  for (TElementType& el : xt::flatten(tensor))
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType>
void save(TArchive& archive, const xt::xarray<TElementType>& tensor)
{
  archive(tensor.dimension());
  for (size_t el : tensor.shape())
  {
    archive(el);
  }
  for (const TElementType& el : xt::flatten(tensor))
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType>
void load(TArchive& archive, xt::xarray<TElementType>& tensor)
{
  size_t rank;
  archive(rank);
  std::vector<size_t> shape(rank);
  for (size_t& el : shape)
  {
    archive(el);
  }
  tensor = xt::xarray<TElementType>(shape);
  for (TElementType& el : xt::flatten(tensor))
  {
    archive(el);
  }
}

} // end of ns cereal
