#include <gtest/gtest.h>

#include "core/types.h"

TEST(SmokeTest, EigenWorks) {
    cpo::VectorXs v(3);
    v << 1.0f, 2.0f, 3.0f;
    EXPECT_FLOAT_EQ(v.sum(), 6.0f);
}

TEST(SmokeTest, EigenMatrixMultiply) {
    cpo::MatrixXs m(2, 2);
    m << 1.0f, 2.0f,
         3.0f, 4.0f;
    cpo::VectorXs v(2);
    v << 1.0f, 1.0f;

    cpo::VectorXs result = m * v;
    EXPECT_FLOAT_EQ(result(0), 3.0f);
    EXPECT_FLOAT_EQ(result(1), 7.0f);
}

TEST(SmokeTest, TypeSizes) {
    EXPECT_EQ(sizeof(cpo::Scalar), 4);     // float = 4 bytes
    EXPECT_EQ(sizeof(cpo::ScalarCPU), 8);  // double = 8 bytes
    EXPECT_EQ(sizeof(cpo::Index), 4);      // int = 4 bytes
}

TEST(SmokeTest, DoublePrecisionEigen) {
    cpo::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    EXPECT_DOUBLE_EQ(v.sum(), 6.0);
}
