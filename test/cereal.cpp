#include <xti/cereal.h>
#include <xti/typedefs.h>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <cereal/archives/binary.hpp>
#include <xtensor/xview.hpp>

TEST_CASE("cereal xfixed")
{
  std::string buffer;

  xti::vec3d v1({1, -0.5, 1e13});
  {
    std::ostringstream stream;
    cereal::BinaryOutputArchive archive(stream);
    archive(v1);
    buffer = stream.str();
  }
  {
    xti::vec3d v2;
    std::istringstream stream;
    stream.str(buffer);
    cereal::BinaryInputArchive archive(stream);
    archive(v2);
    REQUIRE(v1 == v2);
  }
}

TEST_CASE("cereal xtensor")
{
  std::string buffer;

  xt::xtensor<double, 2> m1 = xt::view(xt::linspace<double>(1.0, 10.0, 3), xt::all(), xt::newaxis()) * 5 + xt::view(xt::linspace<double>(1.0, 10.0, 4), xt::newaxis(), xt::all());
  {
    std::ostringstream stream;
    cereal::BinaryOutputArchive archive(stream);
    archive(m1);
    buffer = stream.str();
  }
  {
    xt::xtensor<double, 2> m2;
    std::istringstream stream;
    stream.str(buffer);
    cereal::BinaryInputArchive archive(stream);
    archive(m2);
    REQUIRE(m1 == m2);
  }
}

TEST_CASE("cereal xarray")
{
  std::string buffer;

  xt::xarray<double> m1 = xt::view(xt::linspace<double>(1.0, 10.0, 3), xt::all(), xt::newaxis()) * 5 + xt::view(xt::linspace<double>(1.0, 10.0, 4), xt::newaxis(), xt::all());
  {
    std::ostringstream stream;
    cereal::BinaryOutputArchive archive(stream);
    archive(m1);
    buffer = stream.str();
  }
  {
    xt::xarray<double> m2;
    std::istringstream stream;
    stream.str(buffer);
    cereal::BinaryInputArchive archive(stream);
    archive(m2);
    REQUIRE(m1 == m2);
  }
}
