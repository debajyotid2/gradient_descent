/* Tests for module stats.h
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

#include <math.h>
#include <catch2/catch_test_macros.hpp>
#include <matrix.h>
#include "../src/stats.h"

// Round a double to a given number of decimal places
double round(double num, unsigned int num_dec)
{
    if (num_dec > 6)
        return num;
    long unsigned int fac = 1;
    for (size_t i=0; i<num_dec; i++)
        fac *= 10;
    num *= fac;
    num = floor(num+0.5);
    num /= fac;
    return num;
}

TEST_CASE("Statistical functions.", "[stats]")
{
    Matrix mymat = mat_create(10, 20);

    SECTION("Finding the mean of a matrix along rows.")
    {
        mat_fill(&mymat, -342.23);
        Matrix row_mean = stats_mean(&mymat, 0);

        for (size_t i=0; i<row_mean.nrows; i++)
            for (size_t j=0; j<row_mean.ncols; j++)
                REQUIRE(row_mean.data[i*row_mean.ncols+j]==-342.23);

        mat_destroy(&row_mean);
    }

    SECTION("Finding the mean of a matrix along columns.")
    {
        mat_fill(&mymat, -342.23);
        Matrix col_mean = stats_mean(&mymat, 1);

        for (size_t i=0; i<col_mean.nrows; i++)
            for (size_t j=0; j<col_mean.ncols; j++)
                REQUIRE(round(col_mean.data[i*col_mean.ncols+j], 2)==-342.23);

        mat_destroy(&col_mean);
    }

    SECTION("Finding the mean of a matrix along both axes.")
    {
        mat_fill(&mymat, -342.23);
        Matrix mean = stats_mean(&mymat, 2);

        for (size_t i=0; i<mean.nrows; i++)
            for (size_t j=0; j<mean.ncols; j++)
                REQUIRE(round(mean.data[i*mean.ncols+j], 2)==-342.23);

        mat_destroy(&mean);
    }

    mat_destroy(&mymat);
}
