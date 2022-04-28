#include <xti/eigen.h>
#include <catch2/catch_test_macros.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

TEST_CASE("eigen matrix interface")
{
  xt::xarray<int32_t> m = xt::view(xt::linspace<double>(1.0, 10.0, 3), xt::all(), xt::newaxis()) * 5 + xt::view(xt::linspace<double>(1.0, 10.0, 4), xt::newaxis(), xt::all());

  auto eigen_map = xti::to_eigen(m);
  auto eigen_matrix = xti::to_eigen(decltype(m)(m));
  auto m2_ref = xti::from_eigen(eigen_map);
  auto m2_val = xti::from_eigen(std::move(xti::to_eigen(decltype(m)(m))));

  REQUIRE(eigen_map.data() == &m());
  REQUIRE(eigen_matrix.data() != &m());
  REQUIRE(&m2_ref() == &m());
  REQUIRE(&m2_val() != &m());

  REQUIRE(m == m2_ref);
  REQUIRE(m == m2_val);
  REQUIRE(m == xti::from_eigen(eigen_map));
  REQUIRE(eigen_map.isApprox(eigen_matrix));
  REQUIRE(eigen_map.isApprox(xti::to_eigen(xti::from_eigen(eigen_map))));
  REQUIRE(eigen_map.isApprox(xti::to_eigen(1 * m)));
}
