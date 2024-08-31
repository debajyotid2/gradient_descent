/* Tests for module helpers.h

                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

    Copyright (C) 2024 Debajyoti Debnath

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include "../src/matrix.h"
#include "../src/helpers.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Helpers for linear regression.", "[helpers]")
{
    unsigned int n_features = 20, n_samples = 200;
    unsigned int seed = 2000;
    
    Matrix x = mat_create(n_samples, n_features);
    Matrix y = mat_create(n_samples, 1);
    
    SECTION("Creating linear regression dataset.")
    {
        make_regression_dataset(&x, &y, 1.0, 0.0, seed);
        REQUIRE(x.nrows == n_samples);
        REQUIRE(x.ncols == n_features);
        REQUIRE(y.nrows == n_samples);
        REQUIRE(y.ncols == 1);
    }

    mat_destroy(&x);
    mat_destroy(&y);
}
