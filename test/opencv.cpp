#include <xti/opencv.h>
#include <catch2/catch_test_macros.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>

TEST_CASE("opencv cv::Mat")
{
  xt::xarray<int32_t> m = xt::view(
      xt::view(xt::linspace<double>(1.0, 10.0, 3), xt::all(), xt::newaxis()) * 5 + xt::view(xt::linspace<double>(1.0, 10.0, 4), xt::newaxis(), xt::all()),
      xt::all(), xt::all(), xt::newaxis()
  );

  cv::Mat cv1 = xti::to_opencv(m);
  REQUIRE((void*) cv1.data == (void*) &m());
  REQUIRE(m == xti::from_opencv<int32_t>(cv1));

  const cv::Mat cv2 = xti::to_opencv(m);
  REQUIRE(m == xti::from_opencv<int32_t>(cv2));

  REQUIRE(m == xti::from_opencv<int32_t>(xti::to_opencv(m)));
  REQUIRE(m == xti::from_opencv<int32_t>(const_cast<const cv::Mat&&>(xti::to_opencv(m))));

  REQUIRE(cv::countNonZero(xti::to_opencv(const_cast<const decltype(m)&>(m)) != cv2) == 0);
  REQUIRE(cv::countNonZero(xti::to_opencv(2 * m / 2) != cv2) == 0);
  REQUIRE(cv::countNonZero(xti::to_opencv(decltype(m)(m)) != cv2) == 0);
}

TEST_CASE("opencv cv::imread")
{
  cv::Mat cv = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
  REQUIRE(cv.data);
  auto m = xti::from_opencv<uint8_t>(cv);

  REQUIRE(xt::adapt(m.shape()) == xti::vec3s({IMAGE_HEIGHT, IMAGE_WIDTH, 3}));
}
